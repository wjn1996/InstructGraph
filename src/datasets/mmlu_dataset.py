# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


class MMLUDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split="train", max_length=2048):
        
        with open(split, "r", encoding="utf-8") as fr:
            self.ann = [json.loads(i.strip()) for i in tqdm(fr.readlines())]

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss


        ann = self.ann[index]
        # if ann.get("input", "") == "":
        #     prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        # else:
        #     prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        prompt = ann.get("instruction").strip()
        assert len(ann["answer"][0]) > 0
        example = prompt + ann["answer"][0]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        assert torch.sum((labels!=-100).long()).tolist() > 0

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }
