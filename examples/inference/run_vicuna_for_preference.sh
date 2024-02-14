export CUDA_VISIBLE_DEVICES=7

# ==== Graph Preference Corpus ====

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
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6031  examples/inference/preference_test.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/vicuna-7b-v1.5 \
# --inference_dataset instructgraph_preference_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/preference_tuning/vicuna/instructgraph_hallucination_predictions


# ==== TruthfulQA ====

## inference with only llama2 / vicuna
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6018  examples/inference/preference_test.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/vicuna-7b-v1.5 \
# --inference_dataset truthfulqa_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/preference_tuning/vicuna/truthfulqa_hallucination_predictions


# ==== Anthropic-HH ====

## inference with only llama2 / vicuna
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6032  examples/inference/preference_test.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/vicuna-7b-v1.5 \
# --inference_dataset anthropic_hh_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/preference_tuning/vicuna/anthropic_hh_hallucination_predictions

# ==== HaluEval ====

## inference with only llama2 / vicuna
torchrun --nnodes 1 --nproc_per_node 1 --master_port 6032  examples/inference/preference_test.py \
--model_name_or_path /home/jiw203/wjn/pre-trained-lm/vicuna-7b-v1.5 \
--inference_dataset halueval_dataset \
--inference_save_dir /home/jiw203/wjn/InstructGraph/output/preference_tuning/vicuna/halueval_hallucination_predictions
