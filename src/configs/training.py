# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass, field
from typing import Dict



@dataclass
class train_config:
    model_name: str="PATH/to/LLAMA/7B"
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=8
    batching_strategy: str="packing" #alternative: padding
    context_length: int=2048
    gradient_accumulation_steps: int=1
    num_epochs: int=2
    num_workers_dataloader: int=1
    lr: float=5e-5
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "instructgraph_dataset"
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = "/home/jiw203/wjn/InstructGraph/output/peft"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="/home/jiw203/wjn/InstructGraph/output/fsdp" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    concatedataset_dir: str = "./cache/instructgraph"
    overwrite_concatedataset: bool = False

@dataclass
class aligning_config:
    model_name: str="PATH/to/LLAMA/7B"
    loss_config: Dict[str, str] = field(
        default_factory=lambda: {
            "name": "dpo",
            "beta": 0.1,
            "reference_free": False,
            "label_smoothing": 0,
        }
    )
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=2
    batching_strategy: str="packing" #alternative: padding
    context_length: int=2048
    gradient_accumulation_steps: int=4
    num_epochs: int=1
    num_workers_dataloader: int=1
    lr: float=5e-7
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "instructgraph_preference_dataset"
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=True
    peft_model: str = "" # trained peft model path
    output_dir: str = "/home/jiw203/wjn/InstructGraph/output/peft"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="/home/jiw203/wjn/InstructGraph/output/fsdp" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    concatedataset_dir: str = "./cache/instructgraph"
    overwrite_concatedataset: bool = False

