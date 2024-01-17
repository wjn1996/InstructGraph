# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class InstructGraphDataset(Dataset):
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

class PreferenceGraphDataset(Dataset):
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
        chosen = ann.get("answer_positive")[0]
        rejected = ann.get("answer_negative")[0]
        # assert len(ann["answer_positive"][0]) > 0 and len(ann["answer_negative"][0]) > 0


        chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
        rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)

        assert self.tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
        assert self.tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
        assert self.tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"

        chosen_tokens['input_ids'].append(self.tokenizer.eos_token_id)
        chosen_tokens['attention_mask'].append(1)

        rejected_tokens['input_ids'].append(self.tokenizer.eos_token_id)
        rejected_tokens['attention_mask'].append(1)

        # longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

        # if combined sequence is too long, truncate the prompt
        # if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        #     if truncation_mode == 'keep_start':
        #         prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        #     elif truncation_mode == 'keep_end':
        #         prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        #     else:
        #         raise ValueError(f'Unknown truncation mode: {truncation_mode}')

        # if that's still too long, truncate the response
        # if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        #     chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        #     rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

        # Create labels
        chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
        rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
        chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
        chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
        rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
        rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

        batch = {}

        # batch['prompt'] = prompt
        # batch['chosen'] = prompt + chosen
        # batch['rejected'] = prompt + rejected
        # batch['chosen_response_only'] = chosen
        # batch['rejected_response_only'] = rejected

        """
        batch["chosen_input_ids"]
        batch["rejected_input_ids"]
        batch["prompt_input_ids"]
        batch["chosen_attention_mask"]
        batch["rejected_attention_mask"]
        batch["prompt_attention_mask"]
        batch["chosen_labels"]
        batch["rejected_labels"]
        batch["prompt_labels"]
        """
        # for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens}.items():
            for type_key, tokens in toks.items():
                if type_key == 'token_type_ids':
                    continue
                batch[f'{k}_{type_key}'] = tokens
        
        return batch
    