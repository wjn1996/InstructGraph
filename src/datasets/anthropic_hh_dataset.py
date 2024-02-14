# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


class AnthropicHHDataset(Dataset):
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
        # Create labels
        chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
        rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
        chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
        chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
        rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
        rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

        batch = {}
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