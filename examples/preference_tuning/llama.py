# -*- encoding: utf-8 -*-
'''
@File    :   llama.py
@Time    :   2023/12/28 11:51:30
@Author  :   Jianing Wang
@Contact :   lygwjn@gmail.com
'''

import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

import os
from pkg_resources import packaging

import fire
import random
import torch
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_int8_training, PeftModel
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from src.configs import fsdp_config as FSDP_CONFIG
from src.configs import aligning_config as aligning_config
from src.configs import aligning_config as ALIGN_CONFIG
from src.data.concatenator import ConcatDataset
from src.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from src.utils import fsdp_auto_wrap_policy
from src.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
    get_preference_dataloader_kwargs
)
from src.utils.dataset_utils import get_preprocessed_dataset

from src.utils.train_utils import (
    align,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
    disable_dropout
)

def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model

def unfreeze_peft(model: torch.nn.Module, freeze_module_names: list):
    # unfreeze all parameter at first
    # for _, param in model.named_parameters():
    #     param.requires_grad = True
    # for name, param in model.named_parameters():
    #     name = name.replace("base_model.model.", "")
    #     if name in freeze_module_names:
    #         param.requires_grad = False
    #     if "lora" in name or "lm_head" in name:
    #         param.requires_grad = True
    
    for _, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True

    return model

def print_all_parameters(model: torch.nn):
    print("=======all parameters=====")
    for name, _ in model.named_parameters():
        print(name)
    # for module in model.modules():
    #     print(module)

def main(**kwargs):
    # Update the configuration for the training and sharding process
    aligning_config, fsdp_config = ALIGN_CONFIG(), FSDP_CONFIG()
    update_config((aligning_config, fsdp_config), **kwargs)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(aligning_config.seed)
    torch.manual_seed(aligning_config.seed)
    random.seed(aligning_config.seed)

    if aligning_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # Load the pre-trained model and setup its configuration
    use_cache = False if aligning_config.enable_fsdp else None
    if aligning_config.enable_fsdp and aligning_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            policy = LlamaForCausalLM.from_pretrained(
                aligning_config.model_name,
                load_in_8bit=True if aligning_config.quantization else None,
                device_map="auto" if aligning_config.quantization else None,
                use_cache=use_cache,
                low_cpu_mem_usage=True, 
                torch_dtype="float32"
            )
            reference = LlamaForCausalLM.from_pretrained(
                aligning_config.model_name,
                load_in_8bit=True if aligning_config.quantization else None,
                device_map="auto" if aligning_config.quantization else None,
                use_cache=use_cache,
                low_cpu_mem_usage=True, 
                torch_dtype="float32"
            )
        else:
            llama_config = LlamaConfig.from_pretrained(aligning_config.model_name)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                policy = LlamaForCausalLM(llama_config)
                reference = LlamaForCausalLM(llama_config)

    else:
        policy = LlamaForCausalLM.from_pretrained(
            aligning_config.model_name,
            load_in_8bit=True if aligning_config.quantization else None,
            device_map="auto" if aligning_config.quantization else None,
            use_cache=use_cache,
        )
        reference = LlamaForCausalLM.from_pretrained(
            aligning_config.model_name,
            load_in_8bit=True if aligning_config.quantization else None,
            device_map="auto" if aligning_config.quantization else None,
            use_cache=use_cache,
        )
    
    if aligning_config.enable_fsdp and aligning_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels
        based on the hardware being used. This would speed up fine-tuning.

        this code needs if transformer < 4.36 and torch < 2.1
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            policy = BetterTransformer.transform(policy)
            reference = BetterTransformer.transform(reference)
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(aligning_config.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # print_model_size(model, aligning_config, rank if aligning_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if aligning_config.quantization:
        policy = prepare_model_for_int8_training(policy)
        reference = prepare_model_for_int8_training(reference)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if aligning_config.enable_fsdp and fsdp_config.pure_bf16:
        policy.to(torch.bfloat16)
        reference.to(torch.bfloat16)

    assert aligning_config.use_peft is True, "You must set use_peft is True."
    assert os.path.exists(aligning_config.peft_model) is not None, "You must set use_peft is True."

    peft_config = generate_peft_config(aligning_config, kwargs)
    
    # policy = get_peft_model(model, peft_config)
    # reference = get_peft_model(model, peft_config)
    # policy = PeftModel.from_pretrained(model, aligning_config.peft_model)
    # reference = PeftModel.from_pretrained(model, aligning_config.peft_model)

    # load model with trained lora to form a policy network

    model_module_names = list()
    for name, _ in policy.named_parameters():
        model_module_names.append(name)
    # print("model_module_names=\n", model_module_names)
    
    policy = load_peft_model(policy, aligning_config.peft_model)
    # load model with trained lora to form a reference network
    reference = load_peft_model(reference, aligning_config.peft_model)

    # disable_dropout(policy)
    # disable_dropout(reference)
    policy = unfreeze_peft(policy, model_module_names)
    reference = unfreeze_peft(reference, model_module_names)

    policy.print_trainable_parameters()
    reference.print_trainable_parameters()

    # print_all_parameters(policy)

    #setting up FSDP if enable_fsdp is enabled
    if aligning_config.enable_fsdp:
        if not aligning_config.use_peft and aligning_config.freeze_layers:

            freeze_transformer_layers(aligning_config.num_freeze_layers)

        # mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        # my_auto_wrapping_policy = fsdp_auto_wrap_policy(policy, LlamaDecoderLayer)
        # my_auto_wrapping_reference = fsdp_auto_wrap_policy(reference, LlamaDecoderLayer)
        policy = policy.cuda()
        reference = reference.cuda()
        # policy = FSDP(
        #     policy,
        #     auto_wrap_policy= my_auto_wrapping_policy if aligning_config.use_peft else wrapping_policy,
        #     cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
        #     mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
        #     sharding_strategy=fsdp_config.sharding_strategy,
        #     device_id=torch.cuda.current_device(),
        #     limit_all_gathers=True,
        #     sync_module_states=aligning_config.low_cpu_fsdp,
        #     param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
        #     if aligning_config.low_cpu_fsdp and rank != 0 else None,
        # )

        # reference = FSDP(
        #     reference,
        #     auto_wrap_policy= my_auto_wrapping_reference if aligning_config.use_peft else wrapping_policy,
        #     cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
        #     mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
        #     sharding_strategy=fsdp_config.sharding_strategy,
        #     device_id=torch.cuda.current_device(),
        #     limit_all_gathers=True,
        #     sync_module_states=aligning_config.low_cpu_fsdp,
        #     param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
        #     if aligning_config.low_cpu_fsdp and rank != 0 else None,
        # )

        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(policy)
            apply_fsdp_checkpointing(reference)
    elif not aligning_config.quantization and not aligning_config.enable_fsdp:
        policy.to("cuda")
        reference.to("cuda")

    dataset_config = generate_dataset_config(aligning_config, kwargs)

     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    if not aligning_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="dev",
    )
    if not aligning_config.enable_fsdp or rank == 0:
        print(f"--> Validation Set Length = {len(dataset_val)}")

    if aligning_config.batching_strategy == "packing":
        concatedataset_path = os.path.join(aligning_config.concatedataset_dir, "{}-{}.npy".format(dataset_config.dataset, "train"))
        dataset_train = ConcatDataset(
            dataset_train, 
            chunk_size=aligning_config.context_length, 
            concatedataset_path=concatedataset_path,
            overwrite_concatedataset=aligning_config.overwrite_concatedataset
        )

    train_dl_kwargs = get_preference_dataloader_kwargs(aligning_config, dataset_train, tokenizer, "train")

    # Create DataLoaders for the training and validation dataset
    print("Create DataLoaders for the training dataset ...")
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=aligning_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )

    eval_dataloader = None
    if aligning_config.run_validation:
        if aligning_config.batching_strategy == "packing":
            concatedataset_path = os.path.join(aligning_config.concatedataset_dir, "{}-{}.npy".format(dataset_config.dataset, "validation"))
            dataset_val = ConcatDataset(
                dataset_val, 
                chunk_size=aligning_config.context_length, 
                concatedataset_path=concatedataset_path,
                overwrite_concatedataset=aligning_config.overwrite_concatedataset
            )

        val_dl_kwargs = get_preference_dataloader_kwargs(aligning_config, dataset_val, tokenizer, "val")

        print("Create DataLoaders for the the validation dataset ...")
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=aligning_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            policy.parameters(),
            lr=aligning_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=aligning_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            policy.parameters(),
            lr=aligning_config.lr,
            weight_decay=aligning_config.weight_decay,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=aligning_config.gamma)

    # Start the training process
    results = align(
        policy,
        reference,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        aligning_config.gradient_accumulation_steps,
        aligning_config,
        fsdp_config if aligning_config.enable_fsdp else None,
        local_rank if aligning_config.enable_fsdp else None,
        rank if aligning_config.enable_fsdp else None,
    )
    if not aligning_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    fire.Fire(main)
