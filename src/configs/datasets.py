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