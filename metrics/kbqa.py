# -*- encoding: utf-8 -*-
'''
@File    :   kbqa.py
@Time    :   2023/12/10 16:33:11
@Author  :   Jianing Wang
@Contact :   lygwjn@gmail.com
'''

from tqdm import tqdm
from typing import List
from metrics.metrics import Metrics
from nltk.translate.bleu_score import sentence_bleu


class EntityMatching(Metrics):

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
    

class AnswerMatching(Metrics):

    def __init__(self) -> None:
        super().__init__()
        pass

    @classmethod
    def calculate_item(self, answers: List[str], prediction: str, **kwargs):
        
        metric = {
            "acc": 0,
        }

        prediction = prediction.lower()
        if " is " in prediction:
            prediction = prediction.split(" is ")[1].replace("\"", "").strip()

        is_correct = False
        for answer in answers:
            answer = answer.lower()

            # if "I cannot answer the question directly".lower() in answer and "I cannot answer the question directly".lower() in prediction:
            #     is_correct = True
            #     break

            # if "but the answer is not existing in the graph".lower() in answer and "but the answer is not existing in the graph".lower() in prediction:
            #     is_correct = True
            #     break

            # if "Based on the graph, we can find".lower() in answer and "Based on the graph, we can find".lower() in prediction:
            #     ans = answer.split("so the answer entity is ")[1].strip().replace(".", "")
            #     pred = prediction.split("so the answer entity is ")[1].strip().replace(".", "")
            #     if ans == pred:
            #         is_correct = True
            #         break
            
            answer = answer.split(" is ")[1].replace("\"", "").strip()
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
    