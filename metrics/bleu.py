# -*- encoding: utf-8 -*-
'''
@File    :   bleu.py
@Time    :   2023/12/10 15:59:08
@Author  :   Jianing Wang
@Contact :   lygwjn@gmail.com
'''

from tqdm import tqdm
from typing import List
from metrics.metrics import Metrics
import nltk
from nltk.translate.bleu_score import sentence_bleu
nltk.download('punkt')

class Bleu(Metrics):

    def __init__(self) -> None:
        super().__init__()
        pass
    
    @classmethod
    def calculate_item(self, references: List[str], hypothesis: str, **kwargs):

        metric = {
            "bleu-1": 0,
            "bleu-4": 0,
        }
        references, hypothesis = [i.lower() for i in references], hypothesis.lower()

        tokenizer = kwargs.get("tokenizer")

        # answer = nltk.word_tokenize(answer.lower())
        # prediction = nltk.word_tokenize(prediction.lower())
        references = [tokenizer.tokenize(i) for i in references]
        hypothesis = tokenizer.tokenize(hypothesis)

        bleu_1 = sentence_bleu(references, hypothesis, weights=(0, 0, 0, 0))
        bleu_4 = sentence_bleu(references, hypothesis, weights=(0, 0, 0, 1))

        metric["bleu-1"] += round(bleu_1, 4)
        metric["bleu-4"] += round(bleu_4, 4)

        return metric
    
    @classmethod
    def calculate_metrics(self, examples: list, **kwargs):
        metric = {
            "bleu-1": 0,
            "bleu-4": 0,
        }
        for example in tqdm(examples):
            prediction, answers = example["prediction"], example["answer"]
            
            cur_metric = {
                "bleu-1": 0,
                "bleu-4": 0,
            }
            cur_metric = self.calculate_item(references=answers, hypothesis=prediction, **kwargs)
            metric["bleu-1"] += cur_metric["bleu-1"]
            metric["bleu-4"] += cur_metric["bleu-4"]
        metric["bleu-1"] = round(metric["bleu-1"] / len(examples), 4)
        metric["bleu-4"] = round(metric["bleu-4"] / len(examples), 4)
        return metric