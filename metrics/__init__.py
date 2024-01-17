from metrics.nlgraph import (
    NLGraph_Connectivity, NLGraph_Bipartite, NLGraph_Cycle,
    NLGraph_Degree, NLGraph_HamiltonPath, NLGraph_MaximumFlow,
    NLGraph_ShortestPath, NLGraph_Typology
)
from metrics.bleu import Bleu
from metrics.kbqa import EntityMatching, AnswerMatching
from metrics.cls import Matching, ExactlyMatching
from metrics.graph import (
    GraphMatching, BiGraphMatching, GraphCapacityMatching, 
    GraphWeightMatching, BiGraphWeightMatching, KnowledgeGraphMatching
)

METRICS = {
    "graph-structure-modeling-connectivity-detection": NLGraph_Connectivity,
    "graph-structure-modeling-cycle-detection": NLGraph_Cycle,
    "graph-structure-modeling-maximum-flow": NLGraph_MaximumFlow,
    "graph-structure-modeling-hamilton-path": NLGraph_HamiltonPath,
    "graph-structure-modeling-job-interest": NLGraph_Bipartite,
    "graph-structure-modeling-shortest-path": NLGraph_ShortestPath,
    "graph-structure-modeling-topological-sort": NLGraph_Typology,
    "graph-structure-modeling-degree-computing": NLGraph_Degree,
    "graph-language-modeling-graph-caption-generation-wikipedia": Bleu,
    "graph-language-modeling-graph-caption-generation-webnlg": Bleu,
    "graph-language-modeling-graph-caption-generation-genwiki": Bleu,
    "graph-language-modeling-graph-caption-generation-eventna": Bleu,
    "graph-language-modeling-graph-caption-generation-xalign": Bleu,
    "graph-language-modeling-graph-question-answering-pathquestion": EntityMatching,
    "graph-language-modeling-graph-question-answering-wc2014": EntityMatching,
    "graph-language-modeling-graph-question-answering-grailqa": AnswerMatching,
    "graph-language-modeling-graph-question-answering-webquestions": AnswerMatching,
    "graph-language-modeling-graph-question-answering-wikitablequestions": EntityMatching,
    "graph-language-modeling-graph-node-cls-cora": Matching,
    "graph-language-modeling-graph-node-cls-citeseer": Matching,
    "graph-language-modeling-graph-node-cls-pubmed": Matching,
    "graph-language-modeling-graph-node-cls-ogbn-arxiv": Matching,
    "graph-language-modeling-graph-node-cls-ogbn-products": Matching,
    "graph-language-modeling-graph-link-prediction-wikidata5m": Matching,
    "graph-language-modeling-graph-link-prediction-fb15k237": Matching,
    "graph-language-modeling-graph-link-prediction-conceptnet": Matching,
    "graph-language-modeling-graph-relevance-inspection": ExactlyMatching,
    "graph-language-modeling-graph-collaboration-filtering-amazon": ExactlyMatching,
    "graph-language-modeling-graph-collaboration-filtering-lastfm": ExactlyMatching,
    "graph-construction-modeling-knowledge-graph-generation-wikipedia": KnowledgeGraphMatching,
    "graph-construction-modeling-knowledge-graph-generation-instructionuie": KnowledgeGraphMatching,
    "graph-construction-modeling-knowledge-graph-generation-instructkgc": KnowledgeGraphMatching,
    "graph-construction-modeling-structure-graph-generation-undirected": BiGraphMatching,
    "graph-construction-modeling-structure-graph-generation-undirected2": BiGraphMatching,
    "graph-construction-modeling-structure-graph-generation-undirectedweighted": BiGraphWeightMatching,
    "graph-construction-modeling-structure-graph-generation-directedweighted": GraphCapacityMatching,
    "graph-thought-modeling-natural-language-reasoning-nlpreasoning": None,
    "graph-thought-modeling-factual-knowledge-probing-kbqa": None,

}