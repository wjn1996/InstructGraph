# InstructGraph

This repository is implemented for our paper ```InstructGraph: Empowering Large Language Models to Better Learn from Graph-centric Instruction and Preference``` released in ArXiv.

### What's that?

![](./images/instructgraph_all_task.png)

![](./images/instructgraph_framework.png)

### Released Resource Download


### Quick Start


perform graph instruction tuning by llama2-7b:
```
bash examples/instruction_tuning/run_llama2_flashattn.sh
```

perform inference:
```
bash examples/inference/run_llama2.sh
```

perform evaluation:
```
python3 examples/inference/calculate_metrics.py --inference_save_dir output/instruction_tuning/vicuna/predictions --inference_task graph-language-modeling-graph-question-answering-pathquestion
```


### Demo Play


### Acknowledgement


### Citation