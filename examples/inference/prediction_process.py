# -*- encoding: utf-8 -*-
'''
@File    :   prediction_process.py
@Time    :   2024/01/08 10:21:10
@Author  :   Jianing Wang
@Contact :   lygwjn@gmail.com
'''

"""
不同的backbone在没有demonstration引导下预测的格式各不相同，在此需要对不同的模型进行处理
如果是经过graph instruction后的(is_graph_instruct=True)，基本格式都是符合预期的，因此无需考虑；
如果没有经过任何instruction，即原始的llm，这里预处理的是没有经过instruction的，例如原始的LLAMA、VICUNA。
"""

import os
import json

def process_structure_nlgraph(prediction, model_name):
    return prediction.split("\n")[0]

def process_structure_hamilton(prediction, model_name):
    
    """
    "Yes. The path can be: 0,1,6,3,5,4,2"
    llama 预测结果为
    the path can be:  Yes, there is a path that visits every node exactly once. The path is [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
    需要改成我们的格式，例如
    Yes. The path can be: 0,1,6,3,5,4,2"
    """
    print(prediction)
    prediction = prediction.split("\n")[0]
    if "Yes" in prediction:
        if "The path is [" in prediction:
            return "Yes. The path can be: {}".format(prediction.split("The path is [")[1].replace("].", "").replace(" ", ""))
        elif "The path is: (" in prediction:
            return "Yes. The path can be: {}".format(prediction.split("The path is: (")[1].replace(").", "").replace(" ", "").replace("->", ","))
        else:
            return "No. There is no path."
    else:
        return "No. There is no path."

def process_caption_prediction(prediction, model_name):
    return prediction.split("\n")[0].replace("The verbalization for the triple in the graph is ", "")

def process_caption_wikipedia_prediction(prediction, model_name):
    """
    原始baseline llama在预测caption-wikipedia时候，格式为：

    The encyclopedia passage is:
    ```
    The Netherlands is a country in Western Europe. It is bordered by the North Sea to the north and west, Belgium to the south, and Germany to the east. The Netherlands is a parliamentary democracy organised as a unitary state. The capital city is Amsterdam. The Netherlands is a geographically low-lying country, with about 20% of its area and 21% of its population located below sea level, and polders are common.
    ```

    Note: The answer is a string.

    因此只能取第3行
    """
    try:
        prediction = prediction.split("\n")[2]
    except:
        pass
    return prediction

def process_kbqq(prediction, model_name):
    prediction = "{}".format(prediction.split("\n")[0].strip())
    return prediction

def process_kbqq_grailqa(prediction, model_name):
    prediction = prediction.split("\n")[0].strip()
    # if "is " in prediction:
    #     prediction = prediction.split("is ")[-1].replace(".", "")
    # if "are " in prediction:
    #     prediction = prediction.split("are ")[-1].replace(".", "")
    # if "in the " in prediction:
    #     prediction = prediction.split("in the ")[-1].replace(".", "")
    # if "the " in prediction:
    #     prediction = prediction.split("the ")[-1].replace(".", "")
    return prediction

def process_kbqa_pathquestion(prediction, model_name):
    return prediction.split("\n")[0].strip()

def process_kbqa_wc2014(prediction, model_name):
    prediction = prediction.split("\n")[0].strip().replace(" The reasoning path to answer the question is:", "")
    # if "is " in prediction:
    #     prediction = prediction.split("is ")[-1].replace(".", "")
    # if "from " in prediction:
    #     prediction = prediction.split("from ")[-1].replace(".", "")
    # if "as a " in prediction:
    #     prediction = prediction.split("as a ")[-1].replace(".", "")
    # if "for " in prediction:
    #     prediction = prediction.split("for ")[-1].replace(".", "")
    # if "number " in prediction:
    #     prediction = prediction.split("number ")[-1].replace(".", "")
    # if "team of " in prediction:
    #     prediction = prediction.split("team of ")[-1].replace(".", "")
    # if "team " in prediction:
    #     prediction = prediction.split("team ")[-1].replace(".", "")
    # if "club " in prediction:
    #     prediction = prediction.split("club ")[-1].replace(".", "")
    # if "plays " in prediction:
    #     prediction = prediction.split("plays ")[-1].replace(".", "")
    return prediction

def process_kbqa_wikitablequestions(prediction, model_name):
    return prediction.split("\n")[0].strip()

def process_nodecls(prediction, model_name):
    """
    llama默认输出的结果格式是
    The target scientific publication is classified into 'Neural Networks'.
    """
    prediction = prediction.split("\n")[0].strip().replace("\\_", "_")
    if "is classified into " in prediction:
        prediction = prediction.split("is classified into ")[1].replace("'", "").replace("\"", "")
    if " is " in prediction:
        prediction = prediction.split(" is ")[1].replace("'", "").replace("\"", "")
    if "classified as " in prediction:
        prediction = prediction.split("classified as ")[1].replace("'", "").replace("\"", "")
    if "the category " in prediction:
        prediction = prediction.split("the category ")[1].replace("'", "").replace("\"", "")
    if prediction != "" and prediction[-1] == ".":
        prediction = prediction[:-1]
    return prediction

def process_link(prediction, model_name):
    prediction = prediction.split("\n")[0].strip()
    if " is " in prediction:
        prediction = prediction.split(" is ")[1]
    prediction = prediction.replace("\"", "")
    if prediction != "" and prediction[-1] == ".":
        prediction = prediction[:-1]
    return prediction

def process_relevance(prediction, model_name):
    prediction = prediction.lower()
    if "relevant" in prediction:
        return "Relevant"
    else:
        return "irrelevant"

def process_collaboration(prediction, model_name):
    prediction = prediction.split("\n")[0].strip()
    prediction = prediction.replace("The score is ", "")
    if " is " in prediction:
        prediction = prediction.split(" is ")[-1].strip()
    if prediction != "" and prediction[-1] == ".":
        prediction = prediction[:-1]
    try:
        prediction = str(int(float(prediction)))
    except:
        pass
    return prediction

def process_kgc(prediction, model_name):
    lines = prediction.split("\n")
    start, end = -1, -1
    is_whole = True
    for ei, line in enumerate(lines):
        if "```" in line and start == -1:
            start = ei
            continue
        if "```" in line and start != -1:
            end = ei
            break
    if start != -1 and end != -1:
        prediction = "\n".join(lines[start: end - 1])
    elif start != -1 and end == -1:
        # 没有预测完整，因此最后一行可能是不完整的，因此删除，并补充结尾符号
        if "Graph[name=" in lines[-1]:
            prediction = ""
        elif "entity_list" in lines[-1] or "node_list" in lines[-1]:
            prediction = ""
        elif "triple_list" in lines[-1] or "edge_list" in lines[-1]:
            prediction = ""
        elif lines[-1] == "}":
            prediction += "\n".join(lines[start:-1]) + "\n```"
        else:
            is_whole = False
            prediction = "\n".join(lines[start:-1])[:-1] + "];\n}\n```"
    else:
        prediction = ""
    # if not is_whole:
    #     prediction = prediction[:-1] + "];\n}\n```"
    # print("start=", start)
    # print("end=", end)
    if prediction != "" and prediction[-5:] != "}\n```":
        prediction += "}\n```"
    return prediction



PREDICTION_PROCESSES = {
    "graph-structure-modeling-connectivity-detection": process_structure_nlgraph,
    "graph-structure-modeling-cycle-detection": process_structure_nlgraph,
    "graph-structure-modeling-maximum-flow": process_structure_nlgraph,
    "graph-structure-modeling-hamilton-path": process_structure_hamilton,
    "graph-structure-modeling-job-interest": process_structure_nlgraph,
    "graph-structure-modeling-shortest-path": process_structure_nlgraph,
    "graph-structure-modeling-topological-sort": process_structure_nlgraph,
    "graph-structure-modeling-degree-computing": process_structure_nlgraph,
    "graph-language-modeling-graph-caption-generation-wikipedia": process_caption_wikipedia_prediction,
    "graph-language-modeling-graph-caption-generation-webnlg": process_caption_prediction,
    "graph-language-modeling-graph-caption-generation-genwiki": process_caption_prediction,
    "graph-language-modeling-graph-caption-generation-eventna": process_caption_prediction,
    "graph-language-modeling-graph-caption-generation-xalign": process_caption_prediction,
    "graph-language-modeling-graph-question-answering-pathquestion": process_kbqa_pathquestion,
    "graph-language-modeling-graph-question-answering-wc2014": process_kbqa_wc2014,
    "graph-language-modeling-graph-question-answering-grailqa": process_kbqq_grailqa,
    "graph-language-modeling-graph-question-answering-webquestions": process_kbqq,
    "graph-language-modeling-graph-question-answering-wikitablequestions": process_kbqa_wikitablequestions,
    "graph-language-modeling-graph-node-cls-cora": process_nodecls,
    "graph-language-modeling-graph-node-cls-citeseer": process_nodecls,
    "graph-language-modeling-graph-node-cls-pubmed": process_nodecls,
    "graph-language-modeling-graph-node-cls-ogbn-arxiv": process_nodecls,
    "graph-language-modeling-graph-node-cls-ogbn-products": process_nodecls,
    "graph-language-modeling-graph-link-prediction-wikidata5m": process_link,
    "graph-language-modeling-graph-link-prediction-fb15k237": process_link,
    "graph-language-modeling-graph-link-prediction-conceptnet": process_link,
    "graph-language-modeling-graph-relevance-inspection": process_relevance,
    "graph-language-modeling-graph-collaboration-filtering-amazon": process_collaboration,
    "graph-language-modeling-graph-collaboration-filtering-lastfm": None,
    "graph-construction-modeling-knowledge-graph-generation-wikipedia": process_kgc,
    "graph-construction-modeling-knowledge-graph-generation-instructionuie": process_kgc,
    "graph-construction-modeling-knowledge-graph-generation-instructkgc": process_kgc,
    "graph-construction-modeling-structure-graph-generation-undirected": process_kgc,
    "graph-construction-modeling-structure-graph-generation-undirected2": process_kgc,
    "graph-construction-modeling-structure-graph-generation-undirectedweighted": process_kgc,
    "graph-construction-modeling-structure-graph-generation-directedweighted": process_kgc,
    "graph-thought-modeling-natural-language-reasoning-nlpreasoning": None,
    "graph-thought-modeling-factual-knowledge-probing-kbqa": None,
}


