# -*- encoding: utf-8 -*-
'''
@File    :   preference_test.py
@Time    :   2024/01/22 20:29:43
@Author  :   Jianing Wang
@Contact :   lygwjn@gmail.com
'''

import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

import os
import fire
import torch
import json
import inspect
from random import shuffle
from tqdm import tqdm
from dataclasses import dataclass, field, asdict
from typing import Optional
from peft import PeftModel
import argparse
import transformers
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
from src.utils.dataset_utils import get_preprocessed_dataset
from src.utils.dataset_utils import DATASET_PREPROC
from src.utils.config_utils import update_config
from src.configs import datasets, train_config as TRAIN_CONFIG
from metrics import METRICS
from src.utils.memory_utils import MemoryTrace

from src.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)

print(torch.cuda.device_count())
local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])


# Function to load the main model for text generation
def load_model(model_name, quantization):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model


# Function to load the PeftModel for performance optimization
def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model


# Loading the model from config to load FSDP checkpoints into that
def load_llama_from_config(config_path):
    model_config = LlamaConfig.from_pretrained(config_path) 
    model = LlamaForCausalLM(config=model_config)
    return model


# def eval(model, eval_dataloader, tokenizer, train_config, fsdp_config=None, local_rank=None, rank=None):
#     if train_config.enable_fsdp:
#         world_size = int(os.environ["WORLD_SIZE"])
#     model.eval()
#     eval_preds = []
#     eval_loss = 0.0  # Initialize evaluation loss
#     with MemoryTrace() as memtrace:
#         for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
#             for key in batch.keys():
#                 if train_config.enable_fsdp:
#                     batch[key] = batch[key].to(local_rank)
#                 else:
#                     batch[key] = batch[key].to('cuda:0')
#             # Ensure no gradients are computed for this scope to save memory
#             with torch.no_grad():
#                 # Forward pass and compute loss
#                 outputs = model(**batch)
#                 loss = outputs.loss
#                 eval_loss += loss.detach().float()
#             # Decode predictions and add to evaluation predictions list
#             preds = torch.argmax(outputs.logits, -1)
#             eval_preds.extend(
#                 tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
#             )

#     # If there's more than one CUDA device, reduce evaluation loss across all devices
#     if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
#         dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

#     # Compute average loss and perplexity
#     eval_epoch_loss = eval_loss / len(eval_dataloader)
#     if train_config.enable_fsdp:
#         eval_epoch_loss = eval_epoch_loss/world_size
#     eval_ppl = torch.exp(eval_epoch_loss)

#     # Print evaluation metrics
#     if train_config.enable_fsdp:
#         if local_rank==0:
#             print(f" {eval_ppl=} {eval_epoch_loss=}")
#     else:
#         print(f" {eval_ppl=} {eval_epoch_loss=}")

#     return eval_ppl, eval_epoch_loss


def main(**kwargs):
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument("--model_name_or_path", default="meta-llama/Llama-2-7b-hf", type=str, help="The model name or task of LLaMA / LLaMA2.")
    parser.add_argument("--peft_model", default=None, type=str, help="The peft model.")
    parser.add_argument("--inference_dataset", default=None, type=str, help="The inference dataset name.")
    parser.add_argument("--inference_task", default="all", type=str, help="The inference task name.")
    parser.add_argument("--inference_save_dir", default=None, type=str, help="The saving path.")
    parser.add_argument("--reasoning_type", default="zero-shot", type=str, help="The reasoning type 'zero-shot', 'few-shot-icl', 'few-shot-cot'. 'zero-shot' means directly reasoning based on instruciton, 'few-shot-icl' means use few-shot exemplars with instruction, 'few-shot-cot' means use cot with icl.")
    parser.add_argument("--shot", default=0, type=int, help="The number of exemplars when use few-shot-icl or few-shot-cot")
    parser.add_argument("--sc_num", default=0, type=int, help="Thn self-consistency (SC) prediction ensemble number.")
    args = parser.parse_args()

    # train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()

    if args.inference_task != "all":
        assert args.inference_task in METRICS.keys(), "You must choose one of task in {}".format(METRICS.keys())
    
    if not os.path.exists(args.inference_save_dir):
        os.makedirs(args.inference_save_dir)
        

    # load model
    model = load_model(args.model_name_or_path, quantization=False)
    if args.peft_model is not None:
        model = load_peft_model(model, args.peft_model)
    model.eval()
    # load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset_config = {k:v for k, v in inspect.getmembers(datasets)}[args.inference_dataset]()

    update_config(dataset_config, **kwargs)

    # load dataset
    with open(dataset_config.test_split, "r", encoding="utf-8") as fr:
        inference_dataset = [json.loads(i.strip()) for i in tqdm(fr.readlines())]

    # dataset_eval = get_preprocessed_dataset(
    #     tokenizer,
    #     dataset_config,
    #     split="test",
    # )

    # val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_eval, tokenizer, "val")

    # eval_dataloader = torch.utils.data.DataLoader(
    #         dataset_eval,
    #         num_workers=1,
    #         pin_memory=True,
    #         **val_dl_kwargs,
    #     )
    
    if local_rank == 0:
        print("The preference inference data num: {}".format(len(inference_dataset)))




    inference_saving = list()
    seen_task = list()

    # if os.path.exists(os.path.join(args.inference_save_dir, "prediction.json")):
        # with open(os.path.join(args.inference_save_dir, "prediction.json"), "w", encoding="utf-8") as fw:
        #     inference_saving = json.dump(inference_saving, fw, indent=4)
        # for example in inference_saving:
        #     seen_task.append("{}-{}".format(example["task_name"], example["idx"]))



    num_idx = 0

    for example in tqdm(inference_dataset):
        task_name = example["task_name"]
        if args.inference_task == "all" or args.inference_task == task_name:
            num_idx += 1
            idx = example["idx"]
            if "{}-{}".format(task_name, idx) in seen_task:
                continue
            instruction = example["instruction"]
            answer_positive = example["answer_positive"][0] # list
            answer_negative = example["answer_negative"][0] # list

            chosen_tokens = tokenizer(answer_positive, add_special_tokens=False)
            rejected_tokens = tokenizer(answer_negative, add_special_tokens=False)
            prompt_tokens = tokenizer(instruction, add_special_tokens=False)

            chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
            chosen_tokens['attention_mask'].append(1)

            rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
            rejected_tokens['attention_mask'].append(1)

            chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
            rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
            chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
            chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
            rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
            rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])


            positive_batch = {k: torch.Tensor([v]).long().to("cuda") for k, v in chosen_sequence_tokens.items()}
            negative_batch = {k: torch.Tensor([v]).long().to("cuda") for k, v in rejected_sequence_tokens.items()}

            # print("positive_batch=", positive_batch)

            # negative_batch = tokenizer(negative_instruction, return_tensors="pt")
            # negative_batch = {k: v.to("cuda") for k, v in negative_batch.items()}

            with torch.no_grad():

                positive_loss = model(**positive_batch).loss
                negative_loss = model(**negative_batch).loss

                # print("positive_results=", positive_results)

                # positive_loss = positive_results["loss"][0] if "loss" in positive_results else positive_results[0][0]
                # negative_loss = negative_results["loss"][0] if "loss" in negative_results else negative_results[0][0]

                print("positive_loss=", positive_loss)
                print("negative_loss=", negative_loss)
            prediction = answer_positive if positive_loss < negative_loss else answer_negative
            
            inference_saving.append({
                "task_name": task_name,
                "idx": idx,
                "prediction": prediction,
                "answer_positive": answer_positive,
                "answer_negative": answer_negative,
            })

            if num_idx % 100 == 0:
                with open(os.path.join(args.inference_save_dir, "{}_prediction.json".format(args.inference_task)), "w", encoding="utf-8") as fw:
                    json.dump(inference_saving, fw, indent=4)
    
    print("inference finished, saving all predictions.")
    with open(os.path.join(args.inference_save_dir, "{}_prediction.json".format(args.inference_task)), "w", encoding="utf-8") as fw:
        json.dump(inference_saving, fw, indent=4)
    print("saving finished.")
    

    



if __name__ == "__main__":
    fire.Fire(main)