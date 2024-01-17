# -*- encoding: utf-8 -*-
'''
@File    :   nlgraph.py
@Time    :   2023/12/09 18:30:02
@Author  :   Jianing Wang
@Contact :   lygwjn@gmail.com
'''

from tqdm import tqdm
from typing import List
from metrics.metrics import Metrics
from nltk.translate.bleu_score import sentence_bleu


class NLGraph(Metrics):

    def __init__(self) -> None:
        super().__init__()
        pass

    @classmethod
    def calculate_item(self, answer: str, prediction: str, **kwargs):
        pass
    
    @classmethod
    def calculate_metrics(self, examples: list, **kwargs):
        metric = {
            "acc": 0
        }
        for example in tqdm(examples):
            prediction, answer = example["prediction"], example["answer"]
            cur_metric = self.calculate_item(answer=answer[0], prediction=prediction)
            metric["acc"] += cur_metric["acc"]
        metric["acc"] = round(metric["acc"] / len(examples), 4)
        return metric



class NLGraph_Connectivity(NLGraph):

    def __init__(self) -> None:
        super().__init__()
        pass

    @classmethod
    def calculate_item(self, answer: str, prediction: str, **kwargs):
        answer, prediction = answer.lower(), prediction.lower()
        metric = {
            "acc": 0,
        }
        if "the answer is yes" in answer and "the answer is yes" in prediction:
            metric["acc"] += 1
        elif "the answer is no" in answer and "the answer is no" in prediction:
            metric["acc"] += 1
        elif "the answer is yes" in answer and "yes" in prediction:
            metric["acc"] += 1
        elif "the answer is no" in answer and "yes" not in prediction:
            metric["acc"] += 1
        return metric


class NLGraph_Cycle(NLGraph):

    def __init__(self) -> None:
        super().__init__()
        pass

    @classmethod
    def calculate_item(self, answer: str, prediction: str, **kwargs):
        answer, prediction = answer.lower(), prediction.lower()
        metric = {
            "acc": 0,
        }
        if "there is no cycle" in answer and "there is no cycle" in prediction:
            metric["acc"] += 1
        elif "there is a cycle" in answer and "there is a cycle" in prediction:
            metric["acc"] += 1
        elif "yes" in answer and "yes" in prediction:
            metric["acc"] += 1
        elif "the answer is no" in answer and "yes" not in prediction:
            metric["acc"] += 1
        return metric


class NLGraph_Typology(NLGraph):

    def __init__(self) -> None:
        super().__init__()
        import networkx as nx
    
    @classmethod
    def calculate_item(self, answer: str, prediction: str, **kwargs):

        def check(solution, G):
            n = G.number_of_nodes()
            deg = [0] * n
            for i in range(n):
                deg[i] = G.in_degree[i]
            for node in solution:
                if deg[node] > 0:
                    return 0
                for neighbor in list(G[node]):
                    deg[neighbor] -= 1
            for i in range(n):
                if deg[i] != 0:
                    return 0
            return 1
        
        def process_ans(ans, pos, G):
            num, flag = 0, 0
            solution = []
            n = G.number_of_nodes()
            for i in range(pos, len(ans)):
                if ans[i] >= '0' and ans[i] <='9':
                    num = num*10 + int(ans[i])
                    flag = 1
                else:
                    if flag == 1:
                        solution.append(num)
                        if len(solution) == n:
                            break
                        flag = 0
                    num = 0
            return solution

        def evaluate(ans, G):

            pos = ans.find("solution")
            if pos == -1:
                pos = max(ans.find("yes"), ans.find("in the following order"))
            if pos == -1:
                return 0
            solution = process_ans(ans, pos, G)
            flag1 = check(solution, G)
            solution = process_ans(ans, 0, G)

            flag2 = check(solution, G)
            return (flag1 or flag2)

        answer, prediction = answer.lower(), prediction.lower()
        metric = {
            "acc": 0,
        }
        # waiting for implement

        # waiting for implement
        return metric
    
    @classmethod
    def calculate_metrics(self, examples: list, **kwargs):
        metric = {
            "acc": 0
        }
        for example in tqdm(examples):
            prediction, answer = example["prediction"], example["answer"]
            cur_metric = self.calculate_item(answer=answer[0], prediction=prediction)
            metric["acc"] += cur_metric["acc"]
        metric["acc"] = round(metric["acc"] / len(examples), 4)
        return metric


class NLGraph_ShortestPath(NLGraph):

    def __init__(self) -> None:
        super().__init__()
        pass
    
    @classmethod
    def calculate_item(self, answer: str, prediction: str, **kwargs):

        def get_nodes_and_weight(text: str):
            try:
                weight = text.split(" with a total weight of ")[1].strip().replace(".", "")
            except:
                try:
                    weight = text.split(" = ")[1].strip().replace(".", "")
                except:
                    weight = 0
            try:
                nodes = text.split(" with a total weight of ")[0].strip().split(" is ")[1]
            except:
                try:
                    nodes = text.strip().split(" is ")[1]
                except:
                    nodes = ""
            nodes = set(nodes.split(","))
            return weight, nodes

        metric = {
            "acc": 0,
        }
        answer, prediction = answer.lower(), prediction.lower()
        answer_weight, answer_nodes = get_nodes_and_weight(answer)
        prediction_weight, prediction_nodes = get_nodes_and_weight(prediction)
        if answer_weight == prediction_weight:
            metric["acc"] += 0.5
        metric["acc"] += len(answer_nodes.intersection(prediction_nodes)) / len(answer_nodes.union(prediction_nodes))
        return metric


class NLGraph_MaximumFlow(NLGraph):

    def __init__(self) -> None:
        super().__init__()
        pass
    
    @classmethod
    def calculate_item(self, answer: str, prediction: str, **kwargs):

        def get_res(text: str):
            if " is " in text:
                res = text.split(" is ")[1].strip().replace(".", "")
            else:
                res = text.strip()
            return res

        metric = {
            "acc": 0,
        }
        answer, prediction = answer.lower(), prediction.lower()
        answer_res = get_res(answer)
        prediction_res = get_res(prediction)
        if answer_res == prediction_res:
            metric["acc"] += 1
        return metric


class NLGraph_Bipartite(NLGraph):

    def __init__(self) -> None:
        super().__init__()
        pass

    @classmethod
    def calculate_item(self, answer: str, prediction: str, **kwargs):

        def get_res(text: str):
            text_line = text.split("\n")
            res = list()
            for line in text_line[:-1]:
                try:
                    user = line.split(": ")[0].replace("applicant ", "")
                    item = line.split(": ")[1].replace("job ", "")
                    res.append((user, item))
                except:
                    continue
            return res

        metric = {
            "acc": 0,
        }
        answer, prediction = answer.lower(), prediction.lower()
        answer_res = get_res(answer)
        prediction_res = get_res(prediction)
        answer_dict = dict()
        for ans in answer_res:
            answer_dict[ans[0]] = ans[1]
        correct_num = 0
        for ans in prediction_res:
            if ans[0] in answer_dict.keys() and ans[1] == answer_dict[ans[0]]:
                correct_num += 1
        metric["acc"] += round(correct_num / len(answer_res), 4)
        return metric


class NLGraph_HamiltonPath(NLGraph):

    def __init__(self) -> None:
        super().__init__()
        pass

    @classmethod
    def calculate_item(self, references: str, hypothesis: str, **kwargs):

        metric = {
            "bleu-1": 0,
            "bleu-4": 0,
        }
        references, hypothesis = [i.lower() for i in references], hypothesis.lower()
        references = [answer.split("the path can be: ")[1].split(",") for answer in references]
        try:
            hypothesis = hypothesis.split("the path can be: ")[1].split(",")
        except:
            hypothesis = ""

        bleu_1 = sentence_bleu(references, hypothesis, weights=(1, 0, 0, 0))
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
            cur_metric = self.calculate_item(references=answers, hypothesis=prediction)
            metric["bleu-1"] += cur_metric["bleu-1"]
            metric["bleu-4"] += cur_metric["bleu-4"]
        metric["bleu-1"] = round(metric["bleu-1"] / len(examples), 4)
        metric["bleu-4"] = round(metric["bleu-4"] / len(examples), 4)
        return metric
    

class NLGraph_Degree(NLGraph):

    def __init__(self) -> None:
        super().__init__()
        pass
    
    @classmethod
    def calculate_item(self, answer: str, prediction: str, **kwargs):

        def get_res(text: str):
            if " is " in text:
                res = text.split(" is ")[1].strip().replace(".", "")
            else:
                res = text.strip()
            return res

        metric = {
            "acc": 0,
        }
        answer, prediction = answer.lower(), prediction.lower()
        answer_res = get_res(answer)
        prediction_res = get_res(prediction)
        if answer_res == prediction_res:
            metric["acc"] += 1
        return metric

