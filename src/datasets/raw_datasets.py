# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from datasets import load_dataset
from torch.utils.data import Subset
import re


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):

    def __init__(self, output_path):
        self.output_path = output_path

    def get_data(self):
        return

    def get_instruction(self, sample):
        return

    def get_output(self, sample):
        return

    def get_instruction_and_output(self, sample):
        return


class LocalJsonFileDataset(PromptRawDataset):

    def __init__(self, output_path):
        super().__init__(output_path)
        self.raw_datasets = load_dataset('json', data_files={"data": output_path})

    def get_data(self):
        if self.raw_datasets['data'] is not None:
            return self.raw_datasets['data']
        return None

    def get_instruction(self, sample):
        if sample['instruction'] is not None:
            return sample['instruction']
        return None

    def get_output(self, sample):
        if sample['output'] is not None:
            return sample['output']
        return None

    def get_instruction_and_output(self, sample):
        if sample['instruction'] is not None and sample['output'] is not None:
            return sample['instruction'] + sample['output']
        return None

