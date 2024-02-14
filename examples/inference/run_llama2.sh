export CUDA_VISIBLE_DEVICES=6

# ==== Graph Instruction Corpus ====

## inference for all graph tasks
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6015  examples/inference/llama.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
# --peft_model /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/llama2-peft \
# --inference_dataset instructgraph_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/llama2-peft/predictions

## inference for one task with instruction-version llm
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6054  examples/inference/llama.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
# --peft_model /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/llama2-peft-2epoch \
# --inference_dataset instructgraph_dataset \
# --inference_task graph-construction-modeling-structure-graph-generation-directedweighted \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/llama2-peft-2epoch/predictions

## inference for one task with preference-version llm
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6052  examples/inference/llama.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
# --peft_model /home/jiw203/wjn/InstructGraph/output/preference_tuning/fsdp_peft_flash_1500k/llama2-peft \
# --inference_dataset instructgraph_dataset \
# --inference_task graph-construction-modeling-structure-graph-generation-directedweighted \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/preference_tuning/fsdp_peft_flash_1500k/llama2-peft/predictions


## inference with only llama2 / vicuna
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6017  examples/inference/llama.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-13b-hf \
# --inference_dataset instructgraph_dataset \
# --inference_task graph-structure-modeling-connectivity-detection \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/instruction_tuning/llama2-13B/predictions


# ==== BBH Benchmark ====

## inference for one task with instruction-version llm
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6015  examples/inference/llama.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
# --peft_model /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/llama2-peft-2epoch \
# --inference_dataset big_bench_hard_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/llama2-peft-2epoch/bbh_predictions

# inference with only llama2 / vicuna
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6040  examples/inference/llama.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
# --inference_dataset big_bench_hard_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/instruction_tuning/llama2/bbh_predictions


# ==== MMLU Benchmark ====

## inference for one task with instruction-version llm
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6016  examples/inference/llama.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
# --peft_model /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/llama2-peft-2epoch \
# --inference_dataset mmlu_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/llama2-peft-2epoch/mmlu_predictions

## inference with only llama2 / vicuna
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6020  examples/inference/llama.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
# --inference_dataset mmlu_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/instruction_tuning/llama2/mmlu_predictions


# ==== Planing Benchmark ====

## inference for one task with instruction-version llm
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6060  examples/inference/llama.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
# --peft_model /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/llama2-peft-2epoch \
# --inference_dataset planning_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/llama2-peft-2epoch/planning_predictions

## inference with only llama2 / vicuna
torchrun --nnodes 1 --nproc_per_node 1 --master_port 6061  examples/inference/llama.py \
--model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
--inference_dataset planning_dataset \
--inference_save_dir /home/jiw203/wjn/InstructGraph/output/instruction_tuning/llama2/planning_predictions
