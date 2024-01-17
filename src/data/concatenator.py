# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from tqdm import tqdm
from itertools import chain
import numpy as np

from torch.utils.data import Dataset


class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096, concatedataset_path: str=None, overwrite_concatedataset: bool=False):
        self.dataset = dataset
        self.chunk_size = chunk_size

        if overwrite_concatedataset or not os.path.exists(concatedataset_path):
            self.samples = []
        else:
            print("Loading concatedataset cache from {}".format(concatedataset_path))
            assert overwrite_concatedataset is False and os.path.exists(concatedataset_path)
            self.samples = np.load(concatedataset_path, allow_pickle=True)[()]

        if len(self.samples) == 0:
            buffer = {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
                }

            for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
                buffer = {k: v + sample[k] for k,v in buffer.items()}

                while len(next(iter(buffer.values()))) > self.chunk_size:
                    self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                    buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}
        np.save(concatedataset_path, self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)
