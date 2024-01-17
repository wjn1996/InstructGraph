# -*- encoding: utf-8 -*-
'''
@File    :   collator.py
@Time    :   2023/12/31 13:05:50
@Author  :   Jianing Wang
@Contact :   lygwjn@gmail.com
'''

import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
import torch
from transformers.models.bert import BertTokenizer, BertTokenizerFast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


@dataclass
class PreferenceDataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    This class is used for preference alignment, when the input consists of the following arguments:
    [
        'chosen_input_ids', 'chosen_attention_mask', 'chosen_labels', 
        'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels', 
    ]

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        chosen_labels = [feature["chosen_labels"] for feature in features] if "chosen_labels" in features[0].keys() else None
        rejected_labels = [feature["rejected_labels"] for feature in features] if "rejected_labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if chosen_labels is not None:
            max_label_length = max(len(l) for l in chosen_labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["chosen_labels"]))
                if isinstance(feature["chosen_labels"], list):
                    feature["chosen_labels"] = (
                        feature["chosen_labels"] + remainder if padding_side == "right" else remainder + feature["chosen_labels"]
                    )
                elif padding_side == "right":
                    feature["chosen_labels"] = np.concatenate([feature["chosen_labels"], remainder]).astype(np.int64)
                else:
                    feature["chosen_labels"] = np.concatenate([remainder, feature["chosen_labels"]]).astype(np.int64)

        max_label_length = max(len(l) for l in rejected_labels)
        if self.pad_to_multiple_of is not None:
            max_label_length = (
                (max_label_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        padding_side = self.tokenizer.padding_side
        for feature in features:
            remainder = [self.label_pad_token_id] * (max_label_length - len(feature["rejected_labels"]))
            if isinstance(feature["rejected_labels"], list):
                feature["rejected_labels"] = (
                    feature["rejected_labels"] + remainder if padding_side == "right" else remainder + feature["rejected_labels"]
                )
            elif padding_side == "right":
                feature["rejected_labels"] = np.concatenate([feature["rejected_labels"], remainder]).astype(np.int64)
            else:
                feature["rejected_labels"] = np.concatenate([remainder, feature["rejected_labels"]]).astype(np.int64)

        padding_side = self.tokenizer.padding_side
        for feature in features:
            for feature_key in ["chosen_input_ids", "chosen_attention_mask", "chosen_labels", "rejected_input_ids", "rejected_attention_mask", "rejected_labels"]:
                pad_token_id = 0
                if "input_ids" in feature_key:
                    pad_token_id = self.tokenizer.pad_token_id
                elif "labels" in feature_key:
                    pad_token_id = self.label_pad_token_id
                remainder = [pad_token_id] * (self.max_length - len(feature[feature_key]))
                if isinstance(feature[feature_key], list):
                    feature[feature_key] = feature[feature_key][:self.max_length]
                    feature[feature_key] = (
                        feature[feature_key] + remainder if padding_side == "right" else remainder + feature[feature_key]
                    )
                elif padding_side == "right":
                    feature[feature_key] = feature[feature_key][:self.max_length]
                    feature[feature_key] = np.concatenate([feature[feature_key], remainder]).astype(np.int64)
                else:
                    feature[feature_key] = feature[feature_key][:self.max_length]
                    feature[feature_key] = np.concatenate([remainder, feature[feature_key]]).astype(np.int64)

        # features = self.tokenizer.pad(
        #     features,
        #     padding=self.padding,
        #     max_length=self.max_length,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_tensors=return_tensors,
        # )
        
        features = {k: torch.Tensor([feature[k] if isinstance(feature[k], list) else feature[k].tolist() for feature in features]).long() for k, _ in features[0].items() if not isinstance(features[0][k], str)}

        # prepare decoder_input_ids
        if (
            chosen_labels is not None and rejected_labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            chosen_decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["chosen_labels"])
            rejected_decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["rejected_labels"])
            features["chosen_decoder_input_ids"] = chosen_decoder_input_ids
            features["chosen_decoder_input_ids"] = rejected_decoder_input_ids

        return features
