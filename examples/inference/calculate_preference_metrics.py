# -*- encoding: utf-8 -*-
'''
@File    :   calculate_preference_metrics.py
@Time    :   2024/01/24 21:15:13
@Author  :   Jianing Wang
@Contact :   lygwjn@gmail.com
'''

import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
import os
import numpy as np
import json
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument("--inference_save_dir", default="output/instruction_tuning/fsdp_peft_flash_1500k/vicuna-peft-1epoch/predictions", type=str, help="The saving path.")
    parser.add_argument("--inference_task", default="all", type=str, help="inference task name. 'all' means calculate all task.")
    args = parser.parse_args()

    with open(os.path.join(args.inference_save_dir, "all_prediction.json"), "r", encoding="utf-8") as fr:
        predictions = json.load(fr)
    task_example_num = dict()
    task_acc = dict()
    for example in tqdm(predictions):
        task_name = example["task_name"]
        if args.inference_task is None or args.inference_task == "all" or args.inference_task == task_name:
            if task_name not in task_example_num.keys():
                task_example_num[task_name] = 0
            if task_name not in task_acc.keys():
                task_acc[task_name] = 0
            
            answer_positive = example["answer_positive"] if type(example["answer_positive"]) == str else example["answer_positive"][0]
            prediction = example["prediction"] if type(example["prediction"]) == str else example["prediction"][0]
            if answer_positive == prediction:
                task_acc[task_name] += 1
            task_example_num[task_name] += 1
    
    for task_name in task_example_num.keys():
        print({
            "task_name": task_name,
            "acc": round(task_acc[task_name] / task_example_num[task_name], 4)
        })

if __name__ == "__main__":
    main()


"""
calculate metrics for instructgraph hallucination / preference:
python3 examples/inference/calculate_preference_metrics.py --inference_save_dir output/preference_tuning/llama2/instructgraph_hallucination_predictions --inference_task all




"""