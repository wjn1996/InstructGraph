# -*- encoding: utf-8 -*-
'''
@File    :   cls.py
@Time    :   2023/12/10 17:17:12
@Author  :   Jianing Wang
@Contact :   lygwjn@gmail.com
'''

from tqdm import tqdm
from typing import List
from metrics.metrics import Metrics
from nltk.translate.bleu_score import sentence_bleu


class Matching(Metrics):

    def __init__(self) -> None:
        super().__init__()
        pass
    
    @classmethod
    def calculate_item(self, answers: List[str], prediction: str, **kwargs):
        
        metric = {
            "acc": 0,
        }

        prediction = prediction.lower()

        is_correct = False
        for answer in answers:
            answer = answer.lower()
            if prediction == answer:
                is_correct = True
                break
            if answer in prediction:
                is_correct = True
                break
        
        if is_correct is True:
            metric["acc"] += 1

        return metric
    
    @classmethod
    def calculate_metrics(self, examples: list, **kwargs):

        metric = {
            "acc": 0
        }
        for example in tqdm(examples):
            prediction, answer = example["prediction"], example["answer"]
            cur_metric = self.calculate_item(answers=answer, prediction=prediction)
            metric["acc"] += cur_metric["acc"]
        metric["acc"] = round(metric["acc"] / len(examples), 4)
        return metric


class ExactlyMatching(Metrics):

    def __init__(self) -> None:
        super().__init__()
        pass
    
    @classmethod
    def calculate_item(self, answers: List[str], prediction: str, **kwargs):
        
        metric = {
            "acc": 0,
        }

        prediction = prediction.lower()

        is_correct = False
        for answer in answers:
            answer = answer.lower()
            if prediction == answer:
                is_correct = True
                break
            # if answer in prediction:
            #     is_correct = True
            #     break
        
        if is_correct is True:
            metric["acc"] += 1

        return metric
    
    @classmethod
    def calculate_metrics(self, examples: list, **kwargs):

        metric = {
            "acc": 0
        }
        for example in tqdm(examples):
            prediction, answer = example["prediction"], example["answer"]
            cur_metric = self.calculate_item(answers=answer, prediction=prediction)
            metric["acc"] += cur_metric["acc"]
        metric["acc"] = round(metric["acc"] / len(examples), 4)
        return metric
