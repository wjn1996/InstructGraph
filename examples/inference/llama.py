# -*- encoding: utf-8 -*-
'''
@File    :   llama.py
@Time    :   2023/12/09 11:27:00
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
from src.configs import datasets
from metrics import METRICS

print(torch.cuda.device_count())
local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])


# @dataclass
# class InferenceArgument:
#     """
#     Arguments pertaining to what data we are going to input our model for training and eval.
#     """
#     model_name_or_path: Optional[str] = field(
#         default="meta-llama/Llama-2-7b-hf", metadata={"help": "The model name or task of LLaMA / LLaMA2."}
#     )
#     peft_model: Optional[str] = field(
#         default=None, metadata={"help": "the peft model"}
#     )
#     inference_dataset: Optional[str] = field(
#         default=None, metadata={"help": "the inference dataset name."}
#     )
#     inference_save_dir: Optional[str] = field(
#         default=None, metadata={"help": "the saving path"}
#     )


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
    
    if local_rank == 0:
        print("The inference data num: {}".format(len(inference_dataset)))
    
    inference_saving = list()
    seen_task = list()

    # if os.path.exists(os.path.join(args.inference_save_dir, "prediction.json")):
        # with open(os.path.join(args.inference_save_dir, "prediction.json"), "w", encoding="utf-8") as fw:
        #     inference_saving = json.dump(inference_saving, fw, indent=4)
        # for example in inference_saving:
        #     seen_task.append("{}-{}".format(example["task_name"], example["idx"]))

    exemplars = dict()
    if args.reasoning_type != "zero-shot":
        # 从训练集中随机挑选若干样本作为提示
        print("Random choose some exemplars from training set ...")
        with open(dataset_config.train_split, "r", encoding="utf-8") as fr:
            train_dataset = [json.loads(i.strip()) for i in tqdm(fr.readlines())]
        shuffle(train_dataset)
        # 每个task都随机采样shot个样本作为exemplar
        for example in train_dataset:
            task_name = example["task_name"]
            instruction = example["instruction"]
            answer = example["answer"][0] if args.reasoning_type == "few-shot-icl" else example["answer_with_cot"][0]
            final_prompt = instruction + answer

            if task_name not in exemplars.keys():
                exemplars[task_name] = list()
            if len(exemplars[task_name]) < args.shot:
                exemplars[task_name].append(final_prompt)

    num_idx = 0

    for example in tqdm(inference_dataset):
        task_name = example["task_name"]
        if args.inference_task == "all" or args.inference_task == task_name:
            num_idx += 1
            idx = example["idx"]
            if "{}-{}".format(task_name, idx) in seen_task:
                continue
            instruction = example["instruction"]
            answer = example["answer"]

            if args.reasoning_type != "zero-shot":
                exemplar_list = exemplars[task_name]
                exemplar_prompt = "\n\n".join(exemplar_list)
                instruction = "{}\n\n{}".format(exemplar_prompt, instruction)

            batch = tokenizer(instruction, return_tensors="pt")
            batch = {k: v.to("cuda") for k, v in batch.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **batch,
                    max_new_tokens=200,
                    do_sample=False if args.sc_num == 0 else True,
                    top_p=1.0,
                    temperature=1.0,
                    min_length=None,
                    use_cache=True,
                    top_k=50,
                    repetition_penalty=1.0,
                    length_penalty=1
                )
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            inference_saving.append({
                "task_name": task_name,
                "idx": idx,
                "prediction": output_text,
                "answer": answer
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