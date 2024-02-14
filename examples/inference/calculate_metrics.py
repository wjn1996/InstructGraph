# -*- encoding: utf-8 -*-
'''
@File    :   calculate_metrics.py
@Time    :   2023/12/10 13:52:12
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
from metrics import METRICS
from transformers import LlamaTokenizer
from examples.inference.prediction_process import PREDICTION_PROCESSES


def main():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument("--model_name_or_path", default="meta-llama/Llama-2-7b-hf", type=str, help="The model name or task of LLaMA / LLaMA2.")
    parser.add_argument("--is_graph_instruction", default=False, action="store_true", help="whether the predictions come from instructgraph.")
    parser.add_argument("--inference_save_dir", default="output/instruction_tuning/fsdp_peft_flash_1500k/vicuna-peft-1epoch/predictions", type=str, help="The saving path.")
    parser.add_argument("--inference_task", default="all", type=str, help="inference task name. 'all' means calculate all task.")
    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, token="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")

    with open(os.path.join(args.inference_save_dir, "{}_prediction.json".format(args.inference_task)), "r", encoding="utf-8") as fr:
        predictions = json.load(fr)
    
    predictions_dict = dict()
    metrics = dict()

    for example in tqdm(predictions):
        task_name = example["task_name"]
        prediction = example["prediction"].split("\n")
        start = 0
        for ei in range(len(prediction) - 1, 0, -1):
            if prediction[ei][:2] == "A:":
                start = ei
                break
        # for ei, pred in enumerate(prediction):
        #     if pred[:2] == "A:":
        #         start = ei
        #         break
        final_prediction = "\n".join(prediction[start:])
        if final_prediction[2:] == "A:":
            final_prediction = final_prediction[2:]
        # prediction process
        if not args.is_graph_instruction:
            final_prediction = PREDICTION_PROCESSES[task_name](final_prediction, args.model_name_or_path)

        print(final_prediction)
        print("===")
        example["prediction"] = final_prediction

        if task_name not in predictions_dict.keys():
            predictions_dict[task_name] = list()
        predictions_dict[task_name].append(example)
    
    # print all 
    for task_name, examples in predictions_dict.items():
        if task_name not in METRICS or METRICS[task_name] is None:
            continue
        metric = METRICS[task_name].calculate_metrics(examples, tokenizer=tokenizer)
        # print("task_name: {}, all num: {}, metric: {}".format(task_name, len(examples), metric))
        metrics[task_name] = {
            "example_num": len(examples),
            "metric": metric,
        }
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    main()


"""
command：

If calculate the metric for baseline (e.g., llama)
```
python3 examples/inference/calculate_metrics.py --model_name_or_path meta-llama/Llama-2-7b-hf --inference_save_dir output/instruction_tuning/llama2/predictions --inference_task xxx
```

If calculate the metric for our method (instruction-tuning on graph data), need add "--is_graph_instruction"
```
python3 examples/inference/calculate_metrics.py --model_name_or_path meta-llama/Llama-2-7b-hf --inference_save_dir output/instruction_tuning/fsdp_peft_flash_1500k/llama2-peft-2epoch/predictions --is_graph_instruction --inference_task xxx
```


test for gpt3.5 / gpt4
python3 examples/inference/calculate_metrics.py --is_graph_instruction --inference_save_dir output/openai/gpt4 --inference_task all
"""