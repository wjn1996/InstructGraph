# -*- encoding: utf-8 -*-
'''
@File    :   graph.py
@Time    :   2023/12/11 10:06:11
@Author  :   Jianing Wang
@Contact :   lygwjn@gmail.com
'''

from tqdm import tqdm
from typing import List
from metrics.metrics import Metrics


class GraphMatching(Metrics):
    # 单向图匹配

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def calculate_item(self, answer: str, prediction: str, **kwargs):
        """ answer / prediction example:
        ```
        Graph[name="knowledge-graph"] {
            entity_list = ["0", "1", "2"];
            triple_list = [("0" -> "1"),("0" -> "2")];
        }
        ```
        """

        def get_nodes_and_triples(graph_text: str):
            lines = graph_text.split("\n")
            ### locate gcl
            status = 0
            start, end = 0, len(lines) - 1
            if "```" not in lines[0]:
                lines = ["```"] + lines
            for ei, line in enumerate(lines):
                if line == "```":
                    if status == 0:
                        status = 1
                        start = ei
                    elif status == 1:
                        status = 2
                        end = ei
            gcl = lines[start: end + 1]
            ### find entity list and triple list
            if "entity_list" in graph_text:
                try:
                    entity_list = eval(gcl[2][4:].replace("entity_list = ", "")[:-1])
                except:
                    entity_list = [i[1:-2] if i[0] == "\"" and i[-1] == "\"" else i for i in gcl[2][4:].replace("entity_list = ", "")[1:-2].split(", ")]
            elif "node_list" in graph_text:
                try:
                    entity_list = eval(gcl[2][4:].replace("node_list = ", "")[:-1])
                except:
                    entity_list = [i[1:-2] if i[0] == "\"" and i[-1] == "\"" else i for i in gcl[2][4:].replace("node_list = ", "")[1:-2].split(", ")]
            triple_list = list()
            if "triple_list" in graph_text:
                triple_strs = gcl[3][4:].replace("triple_list = ", "")[:-1]
            elif "edge_list" in graph_text:
                triple_strs = gcl[3][4:].replace("edge_list = ", "")[:-1]
            triple_strs = triple_strs.split("), (")
            # print(triple_strs)
            for triple_str in triple_strs:
                try:
                    head, tail = triple_str.split(" -> ")
                    head = head.replace("[(", "")
                    if head[0] == "\"":
                        head = head[1:]
                    if head[-1] == "\"":
                        head = head[:-1]
                    if tail[0] == "\"":
                        tail = tail[1:]
                    if tail[-1] == "\"":
                        tail = tail[:-1]
                    
                    triple_list.append((head, tail))
                except:
                    continue
            return entity_list, triple_list

        def triple_match(answer_triple_list, prediction_triple_list):
            ground = dict()
            sum = 0
            acc = 0
            precis = 0
            recall = 0
            for head, tail in answer_triple_list:
                ground["{} -> {}".format(head, tail)] = 1
                sum += 1
            for head, tail in prediction_triple_list:
                k = "{} -> {}".format(head, tail)
                if k not in ground.keys():
                    sum += 1
                    continue
                acc += 1.0
            accuracy = round(acc / max(sum, 1), 4)
            precis = round(acc / max(len(prediction_triple_list), 1), 4)
            recall = round(acc / max(len(answer_triple_list), 1), 4)
            return accuracy, precis, recall
            
            

        answer, prediction = answer.lower(), prediction.lower()
        answer_entity_list, answer_triple_list = get_nodes_and_triples(answer)
        prediction_entity_list, prediction_triple_list = get_nodes_and_triples(prediction)

        answer_entity_set = set(answer_entity_list)
        prediction_entity_set = set(prediction_entity_list)

        ner_acc = round(len(answer_entity_set.intersection(prediction_entity_set)) / len(answer_entity_set.union(prediction_entity_set)), 4)
        ner_precis = round(len(answer_entity_set.intersection(prediction_entity_set)) / max(len(prediction_entity_set), 1), 4)
        ner_recall = round(len(answer_entity_set.intersection(prediction_entity_set)) / max(len(answer_entity_set), 1), 4)
        re_acc, re_precis, re_recall = triple_match(answer_triple_list, prediction_triple_list)

        lambda_ = 0.5
        precision = lambda_ * ner_precis + (1 - lambda_) * re_precis
        recall = lambda_ * ner_recall + (1 - lambda_) * re_recall

        return {
            "acc": lambda_ * ner_acc + (1 - lambda_) * re_acc,
            "precision": precision,
            "recall": recall,
            "f1": 2*precision*recall / (precision + recall) if precision + recall != 0 else 0,
            "ner_acc": ner_acc,
            "ner_precis": ner_precis,
            "ner_recall": ner_recall,
            "ner_f1": 2*ner_precis*ner_recall / (ner_precis + ner_recall) if ner_precis + ner_recall != 0 else 0,
            "re_acc": re_acc,
            "re_precis": re_precis,
            "re_recall": re_recall,
            "re_f1": 2*re_precis*re_recall / (re_precis + re_recall) if re_precis + re_recall != 0 else 0,
        }
    
    @classmethod
    def calculate_metrics(self, examples: list, **kwargs):

        metric = {
            "acc": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "ner_acc": 0,
            "ner_precis": 0,
            "ner_recall": 0,
            "ner_f1": 0,
            "re_acc": 0,
            "re_precis": 0,
            "re_recall": 0,
            "re_f1": 0,
        }
        for example in tqdm(examples):
            prediction, answer = example["prediction"], example["answer"]
            cur_metric = self.calculate_item(answer=answer[0], prediction=prediction)
            metric["acc"] += cur_metric["acc"]
            metric["precision"] += cur_metric["precision"]
            metric["recall"] += cur_metric["recall"]
            metric["f1"] += cur_metric["f1"]
            metric["ner_acc"] += cur_metric["ner_acc"]
            metric["ner_precis"] += cur_metric["ner_precis"]
            metric["ner_recall"] += cur_metric["ner_recall"]
            metric["ner_f1"] += cur_metric["ner_f1"]
            metric["re_acc"] += cur_metric["re_acc"]
            metric["re_precis"] += cur_metric["re_precis"]
            metric["re_recall"] += cur_metric["re_recall"]
            metric["re_f1"] += cur_metric["re_f1"]
        
        metric["acc"] = round(metric["acc"] / len(examples), 4)
        metric["precision"] = round(metric["precision"] / len(examples), 4)
        metric["recall"] = round(metric["recall"] / len(examples), 4)
        metric["f1"] = round(metric["f1"] / len(examples), 4)
        metric["ner_acc"] = round(metric["ner_acc"] / len(examples), 4)
        metric["ner_precis"] = round(metric["ner_precis"] / len(examples), 4)
        metric["ner_recall"] = round(metric["ner_recall"] / len(examples), 4)
        metric["ner_f1"] = round(metric["ner_f1"] / len(examples), 4)
        metric["re_acc"] = round(metric["re_acc"] / len(examples), 4)
        metric["re_precis"] = round(metric["re_precis"] / len(examples), 4)
        metric["re_recall"] = round(metric["re_recall"] / len(examples), 4)
        metric["re_f1"] = round(metric["re_f1"] / len(examples), 4)
        return metric


class BiGraphMatching(GraphMatching):
    # 双向图匹配

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def calculate_item(self, answer: str, prediction: str, **kwargs):
        """ answer / prediction example:
        ```
        Graph[name="knowledge-graph"] {
            entity_list = ["0", "1", "2"];
            triple_list = [("0" <-> "1"),("0" <-> "2")];
        }
        ```
        """

        def get_nodes_and_triples(graph_text: str):
            lines = graph_text.split("\n")
            ### locate gcl
            status = 0
            start, end = 0, len(lines) - 1
            if "```" not in lines[0]:
                lines = ["```"] + lines
            for ei, line in enumerate(lines):
                if line == "```":
                    if status == 0:
                        status = 1
                        start = ei
                    elif status == 1:
                        status = 2
                        end = ei
            gcl = lines[start: end + 1]
            ### find entity list and triple list
            if "entity_list" in graph_text:
                try:
                    entity_list = eval(gcl[2][4:].replace("entity_list = ", "")[:-1])
                except:
                    entity_list = [i[1:-2] if i[0] == "\"" and i[-1] == "\"" else i for i in gcl[2][4:].replace("entity_list = ", "")[1:-2].split(", ")]
            elif "node_list" in graph_text:
                try:
                    entity_list = eval(gcl[2][4:].replace("node_list = ", "")[:-1])
                except:
                    try:
                        entity_list = [i[1:-2] if i[0] == "\"" and i[-1] == "\"" else i for i in gcl[2][4:].replace("node_list = ", "")[1:-2].split(", ")]
                    except:
                        entity_list = list()
            else:
                entity_list = list()
            triple_list = list()
            if "triple_list" in graph_text:
                triple_strs = gcl[3][4:].replace("triple_list = ", "")[:-1]
            elif "edge_list" in graph_text:
                try:
                    triple_strs = gcl[3][4:].replace("edge_list = ", "")[:-1]
                except:
                    triple_strs = ""
            else:
                triple_strs = ""
            triple_strs = triple_strs.split("), (")
            # print(triple_strs)
            for triple_str in triple_strs:
                try:
                    head, tail = triple_str.split(" <-> ")
                    head = head.replace("[(", "")
                    if head[0] == "\"":
                        head = head[1:]
                    if head[-1] == "\"":
                        head = head[:-1]
                    if tail[0] == "\"":
                        tail = tail[1:]
                    if tail[-1] == "\"":
                        tail = tail[:-1]
                    
                    triple_list.append((head, tail))
                except:
                    continue
            return entity_list, triple_list

        def triple_match(answer_triple_list, prediction_triple_list):
            ground = dict()
            sum = 0
            acc = 0
            precis = 0
            recall = 0
            for head, tail in answer_triple_list:
                ground["{} -> {}".format(head, tail)] = 1
                sum += 1
            for head, tail in prediction_triple_list:
                k = "{} -> {}".format(head, tail)
                if k not in ground.keys():
                    sum += 1
                    continue
                acc += 1.0
            accuracy = round(acc / max(sum, 1), 4)
            precis = round(acc / max(len(prediction_triple_list), 1), 4)
            recall = round(acc / max(len(answer_triple_list), 1), 4)
            return accuracy, precis, recall
            
            

        answer, prediction = answer.lower(), prediction.lower()
        answer_entity_list, answer_triple_list = get_nodes_and_triples(answer)
        prediction_entity_list, prediction_triple_list = get_nodes_and_triples(prediction)

        answer_entity_set = set(answer_entity_list)
        prediction_entity_set = set(prediction_entity_list)

        ner_acc = round(len(answer_entity_set.intersection(prediction_entity_set)) / len(answer_entity_set.union(prediction_entity_set)), 4)
        ner_precis = round(len(answer_entity_set.intersection(prediction_entity_set)) / max(len(prediction_entity_set), 1), 4)
        ner_recall = round(len(answer_entity_set.intersection(prediction_entity_set)) / max(len(answer_entity_set), 1), 4)
        re_acc, re_precis, re_recall = triple_match(answer_triple_list, prediction_triple_list)

        lambda_ = 0.5
        precision = lambda_ * ner_precis + (1 - lambda_) * re_precis
        recall = lambda_ * ner_recall + (1 - lambda_) * re_recall

        return {
            "acc": lambda_ * ner_acc + (1 - lambda_) * re_acc,
            "precision": precision,
            "recall": recall,
            "f1": 2*precision*recall / (precision + recall) if precision + recall != 0 else 0,
            "ner_acc": ner_acc,
            "ner_precis": ner_precis,
            "ner_recall": ner_recall,
            "ner_f1": 2*ner_precis*ner_recall / (ner_precis + ner_recall) if ner_precis + ner_recall != 0 else 0,
            "re_acc": re_acc,
            "re_precis": re_precis,
            "re_recall": re_recall,
            "re_f1": 2*re_precis*re_recall / (re_precis + re_recall) if re_precis + re_recall != 0 else 0,
        }


class GraphWeightMatching(GraphMatching):
    # 单向带权图
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def calculate_item(self, answer: str, prediction: str, **kwargs):
        """ answer / prediction example:
        ```
        Graph[name="knowledge-graph"] {
            entity_list = ["tunnel", "prisoner"];
            triple_list = [("prisoner" -> "tunnel")[relation="product producer"]];
        }
        ```
        """

        def get_nodes_and_triples(graph_text: str, weight_key: str="weight"):
            lines = graph_text.split("\n")
            ### locate gcl
            status = 0
            start, end = 0, len(lines) - 1
            if "```" not in lines[0]:
                lines = ["```"] + lines
            for ei, line in enumerate(lines):
                if line == "```":
                    if status == 0:
                        status = 1
                        start = ei
                    elif status == 1:
                        status = 2
                        end = ei
            gcl = lines[start: end + 1]
            ### find entity list and triple list
            if "entity_list" in graph_text:
                try:
                    entity_list = eval(gcl[2][4:].replace("entity_list = ", "")[:-1])
                except:
                    entity_list = [i[1:-2] if i[0] == "\"" and i[-1] == "\"" else i for i in gcl[2][4:].replace("entity_list = ", "")[1:-2].split(", ")]
            elif "node_list" in graph_text:
                try:
                    entity_list = eval(gcl[2][4:].replace("node_list = ", "")[:-1])
                except:
                    entity_list = [i[1:-2] if i[0] == "\"" and i[-1] == "\"" else i for i in gcl[2][4:].replace("node_list = ", "")[1:-2].split(", ")]
            triple_list = list()
            if "triple_list" in graph_text:
                triple_strs = gcl[3][4:].replace("triple_list = ", "")[:-1]
            elif "edge_list" in graph_text:
                triple_strs = gcl[3][4:].replace("edge_list = ", "")[:-1]
            triple_strs = [i.split(")[{}=").format(weight_key) for i in triple_strs.split("], (")]
            # print(triple_strs)
            for triple_str in triple_strs:
                try:
                    entity_str, relation = triple_str
                    head, tail = entity_str.split(" -> ")
                    head = head.replace("[(", "")
                    relation = relation.replace("]]", "")
                    if head[0] == "\"":
                        head = head[1:]
                    if head[-1] == "\"":
                        head = head[:-1]
                    if tail[0] == "\"":
                        tail = tail[1:]
                    if tail[-1] == "\"":
                        tail = tail[:-1]
                    if relation[0] == "\"":
                        relation = relation[1:]
                    if relation[-1] == "\"":
                        relation = relation[:-1]
                    triple_list.append((head, relation, tail))
                except:
                    continue
            return entity_list, triple_list

        def triple_match(answer_triple_list, prediction_triple_list):
            ground = dict()
            sum = 0
            acc = 0
            precis = 0
            recall = 0
            for head, rel, tail in answer_triple_list:
                ground["{} -> {}".format(head, tail)] = rel
                sum += 1
            for head, rel, tail in prediction_triple_list:
                k = "{} -> {}".format(head, tail)
                if k not in ground.keys():
                    sum += 1
                    continue
                acc += 0.5 # 能够正确的判断出当前两个实体存在关系
                if ground[k] != rel:
                    continue
                acc += 0.5 # 能够正确识别出两个实体的关系类别
            accuracy = round(acc / max(sum, 1), 4)
            precis = round(acc / max(len(prediction_triple_list), 1), 4)
            recall = round(acc / max(len(answer_triple_list), 1), 4)
            return accuracy, precis, recall
            
        answer, prediction = answer.lower(), prediction.lower()
        answer_entity_list, answer_triple_list = get_nodes_and_triples(answer, weight_key="weight")
        prediction_entity_list, prediction_triple_list = get_nodes_and_triples(prediction, weight_key="weight")

        answer_entity_set = set(answer_entity_list)
        prediction_entity_set = set(prediction_entity_list)

        ner_acc = round(len(answer_entity_set.intersection(prediction_entity_set)) / len(answer_entity_set.union(prediction_entity_set)), 4)
        ner_precis = round(len(answer_entity_set.intersection(prediction_entity_set)) / max(len(prediction_entity_set), 1), 4)
        ner_recall = round(len(answer_entity_set.intersection(prediction_entity_set)) / max(len(answer_entity_set), 1), 4)
        re_acc, re_precis, re_recall = triple_match(answer_triple_list, prediction_triple_list)

        lambda_ = 0.5
        precision = lambda_ * ner_precis + (1 - lambda_) * re_precis
        recall = lambda_ * ner_recall + (1 - lambda_) * re_recall

        return {
            "acc": lambda_ * ner_acc + (1 - lambda_) * re_acc,
            "precision": precision,
            "recall": recall,
            "f1": 2*precision*recall / (precision + recall) if precision + recall != 0 else 0,
            "ner_acc": ner_acc,
            "ner_precis": ner_precis,
            "ner_recall": ner_recall,
            "ner_f1": 2*ner_precis*ner_recall / (ner_precis + ner_recall) if ner_precis + ner_recall != 0 else 0,
            "re_acc": re_acc,
            "re_precis": re_precis,
            "re_recall": re_recall,
            "re_f1": 2*re_precis*re_recall / (re_precis + re_recall) if re_precis + re_recall != 0 else 0,
        }


class GraphCapacityMatching(GraphMatching):
    # 单向带权图
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def calculate_item(self, answer: str, prediction: str, **kwargs):
        """ answer / prediction example:
        ```
        Graph[name="graph"] {
            node_list = ["1", "2"];
            edge_list = [("1" -> "2")[capacity="0"]];
        }
        ```
        """

        def get_nodes_and_triples(graph_text: str, weight_key: str="weight"):
            lines = graph_text.split("\n")
            ### locate gcl
            status = 0
            start, end = 0, len(lines) - 1
            if "```" not in lines[0]:
                lines = ["```"] + lines
            for ei, line in enumerate(lines):
                if line == "```":
                    if status == 0:
                        status = 1
                        start = ei
                    elif status == 1:
                        status = 2
                        end = ei
            gcl = lines[start: end + 1]
            ### find entity list and triple list
            if "entity_list" in graph_text:
                try:
                    entity_list = eval(gcl[2][4:].replace("entity_list = ", "")[:-1])
                except:
                    entity_list = [i[1:-2] if i[0] == "\"" and i[-1] == "\"" else i for i in gcl[2][4:].replace("entity_list = ", "")[1:-2].split(", ")]
            elif "node_list" in graph_text:
                try:
                    entity_list = eval(gcl[2][4:].replace("node_list = ", "")[:-1])
                except:
                    try:
                        entity_list = [i[1:-2] if i[0] == "\"" and i[-1] == "\"" else i for i in gcl[2][4:].replace("node_list = ", "")[1:-2].split(", ")]
                    except:
                        entity_list = list()
            else:
                entity_list = list()
            triple_list = list()
            if "triple_list" in graph_text:
                triple_strs = gcl[3][4:].replace("triple_list = ", "")[:-1]
            elif "edge_list" in graph_text:
                try:
                    triple_strs = gcl[3][4:].replace("edge_list = ", "")[:-1]
                except:
                    triple_strs = ""
            else:
                triple_strs = ""
            triple_strs = [i.split(")[{}=".format(weight_key)) for i in triple_strs.split("], (")]
            for triple_str in triple_strs:
                try:
                    entity_str, relation = triple_str
                    head, tail = entity_str.split(" -> ")
                    head = head.replace("[(", "")
                    relation = relation.replace("]]", "")
                    if head[0] == "\"":
                        head = head[1:]
                    if head[-1] == "\"":
                        head = head[:-1]
                    if tail[0] == "\"":
                        tail = tail[1:]
                    if tail[-1] == "\"":
                        tail = tail[:-1]
                    if relation[0] == "\"":
                        relation = relation[1:]
                    if relation[-1] == "\"":
                        relation = relation[:-1]
                    triple_list.append((head, relation, tail))
                except:
                    continue
            return entity_list, triple_list

        def triple_match(answer_triple_list, prediction_triple_list):
            ground = dict()
            sum = 0
            acc = 0
            precis = 0
            recall = 0
            for head, rel, tail in answer_triple_list:
                ground["{} -> {}".format(head, tail)] = rel
                sum += 1
            print("=====")
            print("ground=", answer_triple_list)
            print("prediction=", prediction_triple_list)
            for head, rel, tail in prediction_triple_list:
                k = "{} -> {}".format(head, tail)
                if k not in ground.keys():
                    sum += 1
                    continue
                acc += 0.5 # 能够正确的判断出当前两个实体存在关系
                if ground[k] != rel:
                    continue
                acc += 0.5 # 能够正确识别出两个实体的关系类别
            accuracy = round(acc / max(sum, 1), 4)
            precis = round(acc / max(len(prediction_triple_list), 1), 4)
            recall = round(acc / max(len(answer_triple_list), 1), 4)
            return accuracy, precis, recall
            
        answer, prediction = answer.lower(), prediction.lower()
        answer_entity_list, answer_triple_list = get_nodes_and_triples(answer, weight_key="capacity")
        prediction_entity_list, prediction_triple_list = get_nodes_and_triples(prediction, weight_key="capacity")
        answer_entity_set = set(answer_entity_list)
        prediction_entity_set = set(prediction_entity_list)

        # print("====")
        # print("answer_entity_set=", answer_entity_set)
        # print("prediction_entity_list=", prediction_entity_list)

        ner_acc = round(len(answer_entity_set.intersection(prediction_entity_set)) / len(answer_entity_set.union(prediction_entity_set)), 4)
        ner_precis = round(len(answer_entity_set.intersection(prediction_entity_set)) / max(len(prediction_entity_set), 1), 4)
        ner_recall = round(len(answer_entity_set.intersection(prediction_entity_set)) / max(len(answer_entity_set), 1), 4)
        re_acc, re_precis, re_recall = triple_match(answer_triple_list, prediction_triple_list)

        lambda_ = 0.5
        precision = lambda_ * ner_precis + (1 - lambda_) * re_precis
        recall = lambda_ * ner_recall + (1 - lambda_) * re_recall

        return {
            "acc": lambda_ * ner_acc + (1 - lambda_) * re_acc,
            "precision": precision,
            "recall": recall,
            "f1": 2*precision*recall / (precision + recall) if precision + recall != 0 else 0,
            "ner_acc": ner_acc,
            "ner_precis": ner_precis,
            "ner_recall": ner_recall,
            "ner_f1": 2*ner_precis*ner_recall / (ner_precis + ner_recall) if ner_precis + ner_recall != 0 else 0,
            "re_acc": re_acc,
            "re_precis": re_precis,
            "re_recall": re_recall,
            "re_f1": 2*re_precis*re_recall / (re_precis + re_recall) if re_precis + re_recall != 0 else 0,
        }



class BiGraphWeightMatching(GraphMatching):
    # 双向带权图
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def calculate_item(self, answer: str, prediction: str, **kwargs):
        """ answer / prediction example:
        ```
        Graph[name="knowledge-graph"] {
            entity_list = ["tunnel", "prisoner"];
            triple_list = [("prisoner" <-> "tunnel")[relation="product producer"]];
        }
        ```
        """

        def get_nodes_and_triples(graph_text: str, weight_key: str="weight"):
            lines = graph_text.split("\n")
            ### locate gcl
            status = 0
            start, end = 0, len(lines) - 1
            if "```" not in lines[0]:
                lines = ["```"] + lines
            for ei, line in enumerate(lines):
                if line == "```":
                    if status == 0:
                        status = 1
                        start = ei
                    elif status == 1:
                        status = 2
                        end = ei
            gcl = lines[start: end + 1]
            ### find entity list and triple list
            if "entity_list" in graph_text:
                try:
                    entity_list = eval(gcl[2][4:].replace("entity_list = ", "")[:-1])
                except:
                    entity_list = [i[1:-2] if i[0] == "\"" and i[-1] == "\"" else i for i in gcl[2][4:].replace("entity_list = ", "")[1:-2].split(", ")]
            elif "node_list" in graph_text:
                try:
                    entity_list = eval(gcl[2][4:].replace("node_list = ", "")[:-1])
                except:
                    try:
                        entity_list = [i[1:-2] if i[0] == "\"" and i[-1] == "\"" else i for i in gcl[2][4:].replace("node_list = ", "")[1:-2].split(", ")]
                    except:
                        entity_list = list()
            else:
                entity_list = list()
            triple_list = list()
            if "triple_list" in graph_text:
                triple_strs = gcl[3][4:].replace("triple_list = ", "")[:-1]
            elif "edge_list" in graph_text:
                try:
                    triple_strs = gcl[3][4:].replace("edge_list = ", "")[:-1]
                except:
                    triple_strs = ""
            else:
                triple_strs = ""
            triple_strs = [i.split(")[{}=".format(weight_key)) for i in triple_strs.split("], (")]

            for triple_str in triple_strs:
                try:
                    entity_str, relation = triple_str
                    head, tail = entity_str.split(" <-> ")
                    head = head.replace("[(", "")
                    relation = relation.replace("]]", "")
                    if head[0] == "\"":
                        head = head[1:]
                    if head[-1] == "\"":
                        head = head[:-1]
                    if tail[0] == "\"":
                        tail = tail[1:]
                    if tail[-1] == "\"":
                        tail = tail[:-1]
                    if relation[0] == "\"":
                        relation = relation[1:]
                    if relation[-1] == "\"":
                        relation = relation[:-1]
                    triple_list.append((head, relation, tail))
                except:
                    continue
            return entity_list, triple_list

        def triple_match(answer_triple_list, prediction_triple_list):
            ground = dict()
            sum = 0
            acc = 0
            precis = 0
            recall = 0
            for head, rel, tail in answer_triple_list:
                ground["{} <-> {}".format(head, tail)] = rel
                sum += 1
            for head, rel, tail in prediction_triple_list:
                k = "{} <-> {}".format(head, tail)
                if k not in ground.keys():
                    sum += 1
                    continue
                acc += 0.5 # 能够正确的判断出当前两个实体存在关系
                if ground[k] != rel:
                    continue
                acc += 0.5 # 能够正确识别出两个实体的关系类别
            accuracy = round(acc / max(sum, 1), 4)
            precis = round(acc / max(len(prediction_triple_list), 1), 4)
            recall = round(acc / max(len(answer_triple_list), 1), 4)
            return accuracy, precis, recall
            
            

        answer, prediction = answer.lower(), prediction.lower()
        answer_entity_list, answer_triple_list = get_nodes_and_triples(answer, weight_key="weight")
        prediction_entity_list, prediction_triple_list = get_nodes_and_triples(prediction, weight_key="weight")
        
        answer_entity_set = set(answer_entity_list)
        prediction_entity_set = set(prediction_entity_list)

        ner_acc = round(len(answer_entity_set.intersection(prediction_entity_set)) / len(answer_entity_set.union(prediction_entity_set)), 4)
        ner_precis = round(len(answer_entity_set.intersection(prediction_entity_set)) / max(len(prediction_entity_set), 1), 4)
        ner_recall = round(len(answer_entity_set.intersection(prediction_entity_set)) / max(len(answer_entity_set), 1), 4)
        re_acc, re_precis, re_recall = triple_match(answer_triple_list, prediction_triple_list)

        lambda_ = 0.5
        precision = lambda_ * ner_precis + (1 - lambda_) * re_precis
        recall = lambda_ * ner_recall + (1 - lambda_) * re_recall

        return {
            "acc": lambda_ * ner_acc + (1 - lambda_) * re_acc,
            "precision": precision,
            "recall": recall,
            "f1": 2*precision*recall / (precision + recall) if precision + recall != 0 else 0,
            "ner_acc": ner_acc,
            "ner_precis": ner_precis,
            "ner_recall": ner_recall,
            "ner_f1": 2*ner_precis*ner_recall / (ner_precis + ner_recall) if ner_precis + ner_recall != 0 else 0,
            "re_acc": re_acc,
            "re_precis": re_precis,
            "re_recall": re_recall,
            "re_f1": 2*re_precis*re_recall / (re_precis + re_recall) if re_precis + re_recall != 0 else 0,
        }


class KnowledgeGraphMatching(GraphMatching):

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def calculate_item(self, answer: str, prediction: str, **kwargs):
        """ answer / prediction example:
        ```
        Graph[name="knowledge-graph"] {
            entity_list = ["tunnel", "prisoner"];
            triple_list = [("prisoner" -> "tunnel")[relation="product producer"]];
        }
        ```
        """

        def get_nodes_and_triples(graph_text: str):
            lines = graph_text.split("\n")
            ### locate gcl
            status = 0
            start, end = 0, len(lines) - 1
            if "```" not in lines[0]:
                lines = ["```"] + lines
            for ei, line in enumerate(lines):
                if line == "```":
                    if status == 0:
                        status = 1
                        start = ei
                    elif status == 1:
                        status = 2
                        end = ei
            gcl = lines[start: end + 1]
            ### find entity list and triple list
            if "entity_list" in graph_text:
                try:
                    entity_list = eval(gcl[2][4:].replace("entity_list = ", "")[:-1])
                except:
                    try:
                        entity_list = [i[1:-2] if i[0] == "\"" and i[-1] == "\"" else i for i in gcl[2][4:].replace("entity_list = ", "")[1:-2].split(", ")]
                    except:
                        entity_list = list()
            elif "node_list" in graph_text:
                try:
                    entity_list = eval(gcl[2][4:].replace("node_list = ", "")[:-1])
                except:
                    entity_list = [i[1:-2] if i[0] == "\"" and i[-1] == "\"" else i for i in gcl[2][4:].replace("node_list = ", "")[1:-2].split(", ")]
            else:
                entity_list = list()
            triple_list = list()
            if "triple_list" in graph_text:
                try:
                    triple_strs = gcl[3][4:].replace("triple_list = ", "")[:-1]
                except:
                    triple_strs = ""
            elif "edge_list" in graph_text:
                triple_strs = gcl[3][4:].replace("edge_list = ", "")[:-1]
            else:
                triple_strs = ""
            triple_strs = [i.split(")[relation=") for i in triple_strs.split("], (")]
            # print(triple_strs)
            for triple_str in triple_strs:
                try:
                    entity_str, relation = triple_str
                    head, tail = entity_str.split(" -> ")
                    head = head.replace("[(", "")
                    relation = relation.replace("]]", "")
                    if head[0] == "\"":
                        head = head[1:]
                    if head[-1] == "\"":
                        head = head[:-1]
                    if tail[0] == "\"":
                        tail = tail[1:]
                    if tail[-1] == "\"":
                        tail = tail[:-1]
                    if relation[0] == "\"":
                        relation = relation[1:]
                    if relation[-1] == "\"":
                        relation = relation[:-1]
                    triple_list.append((head, relation, tail))
                except:
                    continue
            return entity_list, triple_list

        def triple_match(answer_triple_list, prediction_triple_list):
            ground = dict()
            sum = 0
            acc = 0
            precis = 0
            recall = 0
            for head, rel, tail in answer_triple_list:
                ground["{} -> {}".format(head, tail)] = rel
                sum += 1
            for head, rel, tail in prediction_triple_list:
                k = "{} -> {}".format(head, tail)
                if k not in ground.keys():
                    sum += 1
                    continue
                acc += 0.5 # 能够正确的判断出当前两个实体存在关系
                if ground[k] != rel:
                    continue
                acc += 0.5 # 能够正确识别出两个实体的关系类别
            accuracy = round(acc / max(sum, 1), 4)
            precis = round(acc / max(len(prediction_triple_list), 1), 4)
            recall = round(acc / max(len(answer_triple_list), 1), 4)
            return accuracy, precis, recall
            
            

        answer, prediction = answer.lower(), prediction.lower()
        answer_entity_list, answer_triple_list = get_nodes_and_triples(answer)
        prediction_entity_list, prediction_triple_list = get_nodes_and_triples(prediction)

        answer_entity_set = set(answer_entity_list)
        prediction_entity_set = set(prediction_entity_list)

        ner_acc = round(len(answer_entity_set.intersection(prediction_entity_set)) / len(answer_entity_set.union(prediction_entity_set)), 4)
        ner_precis = round(len(answer_entity_set.intersection(prediction_entity_set)) / max(len(prediction_entity_set), 1), 4)
        ner_recall = round(len(answer_entity_set.intersection(prediction_entity_set)) / max(len(answer_entity_set), 1), 4)
        re_acc, re_precis, re_recall = triple_match(answer_triple_list, prediction_triple_list)

        lambda_ = 0.5
        precision = lambda_ * ner_precis + (1 - lambda_) * re_precis
        recall = lambda_ * ner_recall + (1 - lambda_) * re_recall

        return {
            "acc": lambda_ * ner_acc + (1 - lambda_) * re_acc,
            "precision": precision,
            "recall": recall,
            "f1": 2*precision*recall / (precision + recall) if precision + recall != 0 else 0,
            "ner_acc": ner_acc,
            "ner_precis": ner_precis,
            "ner_recall": ner_recall,
            "ner_f1": 2*ner_precis*ner_recall / (ner_precis + ner_recall) if ner_precis + ner_recall != 0 else 0,
            "re_acc": re_acc,
            "re_precis": re_precis,
            "re_recall": re_recall,
            "re_f1": 2*re_precis*re_recall / (re_precis + re_recall) if re_precis + re_recall != 0 else 0,
        }
    
    @classmethod
    def calculate_metrics(self, examples: list, **kwargs):

        metric = {
            "acc": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "ner_acc": 0,
            "ner_precis": 0,
            "ner_recall": 0,
            "ner_f1": 0,
            "re_acc": 0,
            "re_precis": 0,
            "re_recall": 0,
            "re_f1": 0,
        }
        for example in tqdm(examples):
            prediction, answer = example["prediction"], example["answer"]
            cur_metric = self.calculate_item(answer=answer[0], prediction=prediction)
            metric["acc"] += cur_metric["acc"]
            metric["precision"] += cur_metric["precision"]
            metric["recall"] += cur_metric["recall"]
            metric["f1"] += cur_metric["f1"]
            metric["ner_acc"] += cur_metric["ner_acc"]
            metric["ner_precis"] += cur_metric["ner_precis"]
            metric["ner_recall"] += cur_metric["ner_recall"]
            metric["ner_f1"] += cur_metric["ner_f1"]
            metric["re_acc"] += cur_metric["re_acc"]
            metric["re_precis"] += cur_metric["re_precis"]
            metric["re_recall"] += cur_metric["re_recall"]
            metric["re_f1"] += cur_metric["re_f1"]
        
        metric["acc"] = round(metric["acc"] / len(examples), 4)
        metric["precision"] = round(metric["precision"] / len(examples), 4)
        metric["recall"] = round(metric["recall"] / len(examples), 4)
        metric["f1"] = round(metric["f1"] / len(examples), 4)
        metric["ner_acc"] = round(metric["ner_acc"] / len(examples), 4)
        metric["ner_precis"] = round(metric["ner_precis"] / len(examples), 4)
        metric["ner_recall"] = round(metric["ner_recall"] / len(examples), 4)
        metric["ner_f1"] = round(metric["ner_f1"] / len(examples), 4)
        metric["re_acc"] = round(metric["re_acc"] / len(examples), 4)
        metric["re_precis"] = round(metric["re_precis"] / len(examples), 4)
        metric["re_recall"] = round(metric["re_recall"] / len(examples), 4)
        metric["re_f1"] = round(metric["re_f1"] / len(examples), 4)
        return metric
