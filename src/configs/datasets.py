# -*- encoding: utf-8 -*-
'''
@File    :   datasets.py
@Time    :   2023/11/29 10:27:25
@Author  :   Jianing Wang
@Contact :   lygwjn@gmail.com
'''

from dataclasses import dataclass

@dataclass
class instructgraph_dataset:
    dataset: str = "instructgraph_dataset"
    train_split: str = "data/instruction_dataset/released/instructgraph_train_data.json"
    test_split: str = "data/instruction_dataset/released/instructgraph_test_data.json"
    dev_split: str = "data/instruction_dataset/released/instructgraph_test_data_small.json"

class instructgraph_preference_dataset:
    dataset: str = "instructgraph_preference_dataset"
    train_split: str = "data/preference_dataset/released/instructgraph_train_preference_data.json"
    test_split: str = "data/preference_dataset/released/instructgraph_test_preference_data.json"
    dev_split: str = "data/preference_dataset/released/instructgraph_test_preference_data_small.json"

class big_bench_hard_dataset:
    dataset: str = "big_bench_hard"
    train_split: str = ""
    test_split: str = "data/other_benchmarks/BIG-Bench-Hard/bbh_dataset.json"
    dev_split: str = ""

class mmlu_dataset:
    dataset: str = "mmlu"
    train_split: str = ""
    test_split: str = "data/other_benchmarks/MMLU/mmlu_dataset.json"
    dev_split: str = ""

class truthfulqa_dataset:
    dataset: str = "truthfulqa"
    train_split: str = ""
    test_split: str = "data/other_benchmarks/TruthfulQA/truthfulqa_dataset.json"
    dev_split: str = ""

class halueval_dataset:
    dataset: str = "halueval"
    train_split: str = ""
    test_split: str = "data/other_benchmarks/HaluEval/halueval_dataset.json"
    dev_split: str = ""

class anthropic_hh_dataset:
    dataset: str = "anthropic_hh"
    train_split: str = ""
    test_split: str = "data/other_benchmarks/Anthropic-hh-rlhf/anthropic_hh_dataset.json"
    dev_split: str = ""

class planning_dataset:
    dataset: str = "planning"
    train_split: str = ""
    test_split: str = "data/other_benchmarks/Planning/planning_dataset.json"
    dev_split: str = ""

# @dataclass
# class samsum_dataset:
#     dataset: str =  "samsum_dataset"
#     train_split: str = "train"
#     test_split: str = "validation"
    
    
# @dataclass
# class grammar_dataset:
#     dataset: str = "grammar_dataset"
#     train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" 
#     test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"

    
# @dataclass
# class alpaca_dataset:
#     dataset: str = "alpaca_dataset"
#     train_split: str = "train"
#     test_split: str = "val"
#     data_path: str = "src/llama_recipes/datasets/alpaca_data.json"
    
    
# @dataclass
# class custom_dataset:
#     dataset: str = "custom_dataset"
#     file: str = "examples/custom_dataset.py"
#     train_split: str = "train"
#     test_split: str = "validation"