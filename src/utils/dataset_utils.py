# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import importlib
from functools import partial
from pathlib import Path
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
import os
import hashlib
from itertools import chain
# from . import raw_datasets
from src.datasets import raw_datasets


import torch

from src.datasets import (
    get_instructgraph_dataset,
    get_instructgraph_preference_dataset,
    get_grammar_dataset,
    get_alpaca_dataset,
    get_samsum_dataset,
)


def load_module_from_py_file(py_file: str) -> object:
    """
    This method loads a module from a py file which is not in the Python path
    """
    module_name = Path(py_file).name
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)

    loader.exec_module(module)

    return module


def get_custom_dataset(dataset_config, tokenizer, split: str):
    if ":" in dataset_config.file:
        module_path, func_name = dataset_config.file.split(":")
    else:
        module_path, func_name = dataset_config.file, "get_custom_dataset"

    if not module_path.endswith(".py"):
        raise ValueError(f"Dataset file {module_path} is not a .py file.")

    module_path = Path(module_path)
    if not module_path.is_file():
        raise FileNotFoundError(f"Dataset py file {module_path.as_posix()} does not exist or is not a file.")

    module = load_module_from_py_file(module_path.as_posix())
    try:
        return getattr(module, func_name)(dataset_config, tokenizer, split)
    except AttributeError as e:
        print(f"It seems like the given method name ({func_name}) is not present in the dataset .py file ({module_path.as_posix()}).")
        raise e


DATASET_PREPROC = {
    "instructgraph_dataset": partial(get_instructgraph_dataset),
    "instructgraph_preference_dataset": partial(get_instructgraph_preference_dataset),
    "alpaca_dataset": partial(get_alpaca_dataset),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "custom_dataset": get_custom_dataset,
}


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train"
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        if split == "train":
            return dataset_config.train_split
        elif split == "dev":
            return dataset_config.dev_split
        return dataset_config.test_split

    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        get_split(),
    )


class PromptDataset(Dataset):

    def __init__(self, chosen_dataset):
        super().__init__()
        self.chosen_dataset = chosen_dataset

    def __len__(self):
        length = len(self.chosen_dataset)
        return length

    def __getitem__(self, idx):
        return {
                "input_ids": self.chosen_dataset[idx]["input_ids"],
                "attention_mask": self.chosen_dataset[idx]["attention_mask"],
                "labels": self.chosen_dataset[idx]["labels"]
        }


def get_raw_dataset(output_path):
    return raw_datasets.LocalJsonFileDataset(output_path)


def create_dataset_split(current_dataset, raw_dataset, tokenizer, end_of_conversation_token, max_seq_len):
    chosen_dataset = []
    for i, tmp_data in enumerate(current_dataset):
        # tokenize the text
        chosen_sentence = raw_dataset.get_instruction_and_output(tmp_data)
        input_sentence = raw_dataset.get_instruction(tmp_data)
        if chosen_sentence is not None:
            chosen_sentence += end_of_conversation_token
            chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
            input_token = tokenizer(input_sentence,
                                        max_length=max_seq_len,
                                        padding="max_length",
                                        truncation=True,
                                        return_tensors="pt")
            chosen_token["labels"] = chosen_token["input_ids"].clone().detach()
            chosen_token["labels"][input_token["input_ids"][:]==chosen_token["input_ids"][:]] = -100
            chosen_token["input_ids"] = chosen_token["input_ids"].squeeze(0)
            chosen_token["attention_mask"] = chosen_token["attention_mask"].squeeze(0)
            chosen_token["labels"] = chosen_token["labels"].squeeze(0)
            chosen_dataset.append(chosen_token)
    return PromptDataset(chosen_dataset)


def create_dataset(local_rank, train_path, seed, tokenizer, end_of_conversation_token, max_seq_len):
    raw_dataset = get_raw_dataset(train_path)
    train_dataset = raw_dataset.get_data()
    train_dataset = create_dataset_split(train_dataset, raw_dataset, tokenizer,
                                         end_of_conversation_token,
                                         max_seq_len)
    return train_dataset


def create_prompt_dataset(local_rank,
                          train_path,
                          output_path,
                          seed,
                          tokenizer,
                          max_seq_len,
                          end_of_conversation_token="<|endoftext|>",
                          reload=False,
                          is_train=True):
    """
    Creates the prompt dataset
    """
    if is_train:
        name = "traindata.pt"
    else:
        name = "evaldata.pt"
    train_fname = output_path + name
    cache_found = os.path.isfile(train_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).cuda()
    torch.distributed.all_reduce(buf_create_cache)
    if local_rank <= 0 and (buf_create_cache.item() != 0 or reload):
        train_dataset = create_dataset(
                local_rank, train_path, seed, tokenizer, end_of_conversation_token, max_seq_len)
        torch.save(train_dataset, train_fname)
    torch.distributed.barrier()
    return torch.load(train_fname)

