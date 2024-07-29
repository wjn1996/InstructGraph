# InstructGraph

[![arXiv](https://img.shields.io/badge/arXiv-2402.08785-b31b1b.svg)](https://arxiv.org/abs/2402.08785) 
![License](https://img.shields.io/badge/License-MIT-blue)

This repository is implemented for our paper [```InstructGraph: Boosting Large Language Models via Graph-centric Instruction Tuning and Preference Alignment```](https://arxiv.org/pdf/2402.08785.pdf).

🆕 [24-05-16] Our paper has been accepted to the Findings of ACL 2024.

---

### What's InstructGraph?

**InstructGraph** is a framework for empowering large language models (LLMs) on graph-centric tasks via graph instruction tuning and preference alignment. We collect 29 standard graph datasets and decompose them into four groups, including graph structure modeling, graph language modeling, graph generation modeling, and graph thought modeling.

To better bridge the gap between textual LLMs with the graph data, we introduce a **structured format verbalizer**, which aims to transform the graph data into a code-like format. This interface can enable the LLM to reuse the ability of code understanding and generation. In addition, the LLM can generate a graph by outputting a code-like sequence. 

![](./images/instructgraph_all_task.png)

We also explore four hallucination problems in graph reasoning and generation, respectively. We use direct preference optimization (DPO) to perform preference alignment.

![](./images/instructgraph_framework.png)

More details can be found in our paper.

<!-- ### Released Resource Download -->

---

### Quick Start

Download the open-resource [llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf) to a folder, e.g., "./pre-trained-lm/Llama-2-7b-hf".

We release the instruction corpus at: [HuggingFace](https://huggingface.co/datasets/wjn1996/InstructGraph).

**Step1:** Perform graph instruction tuning by llama2-7b with lora:
```
bash examples/instruction_tuning/run_llama2_flashattn.sh
```

You can obtain a resulting folder in "./output/" with two files, i.e., "adapter_config.json" and "adapter_model.bin".

**Step2:** Perform graph preference alignment by llama2-7b with lora:

You must first set the argument "--peft_model" as the folder of instruction tuning checkpoint, and then:
```
bash examples/preference_tuning/run_llama2_flashattn.sh
```

**Step3:** Perform inference on graph instruction tasks:
```
bash examples/inference/run_llama2.sh
```

**Step4:** perform inference on preference task:
```
bash examples/inference/run_llama2_for_preference.sh
```

**Step5:** Calculate metrics on graph instruction tasks, e.g., "graph-language-modeling-graph-question-answering-pathquestion":

```
python3 examples/inference/calculate_metrics.py \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--inference_save_dir output/instruction_tuning/fsdp_peft_flash_1500k/llama2-peft-2epoch/predictions \
--is_graph_instruction \
--inference_task graph-language-modeling-graph-question-answering-pathquestion
```

**Step5:** Calculate metrics on graph preference tasks.

```
python3 examples/inference/calculate_preference_metrics.py \
--inference_save_dir output/preference_tuning/llama2/instructgraph_hallucination_predictions \
--inference_task all
```

### Demo Play

Please see in the jupyter file [instruction.ipynb](./examples/demo/instruction.ipynb).

<!-- ### Acknowledgement -->



### Citation

```
@article{Wang2024InstructGraph,
  author       = {Jianing Wang and
                  Junda Wu and
                  Yupeng Wu and
                  Yao Liu and
                  Ming Gao and
                  Julian McAuley},
  title        = {InstructGraph: Boosting Large Language Models via Graph-centric Instruction Tuning and Preference Alignment},
  eprinttype    = {arXiv},
  eprint       = {2402.08785},
}
```
