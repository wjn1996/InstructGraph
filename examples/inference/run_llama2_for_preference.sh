export CUDA_VISIBLE_DEVICES=3

# ==== Graph Preference Corpus ====

## inference for one task with instruction-version llm
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6051  examples/inference/preference_test.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
# --peft_model /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/llama2-peft-2epoch \
# --inference_dataset instructgraph_preference_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/preference_tuning/fsdp_peft_flash_1500k/llama2-peft-2epoch/instructgraph_hallucination_predictions

## inference for one task with preference-version llm
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6052  examples/inference/preference_test.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
# --peft_model /home/jiw203/wjn/InstructGraph/output/preference_tuning/fsdp_peft_flash_1500k/llama2-peft \
# --inference_dataset instructgraph_preference_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/preference_tuning/fsdp_peft_flash_1500k/llama2-peft-preference-2epoch/instructgraph_hallucination_predictions


## inference with only llama2 / vicuna
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6050  examples/inference/preference_test.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
# --inference_dataset instructgraph_preference_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/preference_tuning/llama2/instructgraph_hallucination_predictions


# ==== TruthfulQA ====

## inference for one task with instruction-version llm
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6016  examples/inference/preference_test.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
# --peft_model /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/llama2-peft-2epoch \
# --inference_dataset truthfulqa_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/preference_tuning/fsdp_peft_flash_1500k/llama2-peft-2epoch/truthfulqa_hallucination_predictions


## inference for one task with preference-version llm
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6017  examples/inference/preference_test.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
# --peft_model /home/jiw203/wjn/InstructGraph/output/preference_tuning/fsdp_peft_flash_1500k/llama2-peft \
# --inference_dataset truthfulqa_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/preference_tuning/fsdp_peft_flash_1500k/llama2-peft-preference-2epoch/truthfulqa_hallucination_predictions


## inference with only llama2 / vicuna
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6015  examples/inference/preference_test.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
# --inference_dataset truthfulqa_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/preference_tuning/llama2/truthfulqa_hallucination_predictions


# ==== Anthropic-HH ====

## inference for one task with instruction-version llm
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6021  examples/inference/preference_test.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
# --peft_model /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/llama2-peft-2epoch \
# --inference_dataset anthropic_hh_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/preference_tuning/fsdp_peft_flash_1500k/llama2-peft-2epoch/anthropic_hh_hallucination_predictions


## inference for one task with preference-version llm
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6031  examples/inference/preference_test.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
# --peft_model /home/jiw203/wjn/InstructGraph/output/preference_tuning/fsdp_peft_flash_1500k/llama2-peft \
# --inference_dataset anthropic_hh_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/preference_tuning/fsdp_peft_flash_1500k/llama2-peft-preference-2epoch/anthropic_hh_hallucination_predictions


## inference with only llama2 / vicuna
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6020  examples/inference/preference_test.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
# --inference_dataset anthropic_hh_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/preference_tuning/llama2/anthropic_hh_hallucination_predictions


# ==== HaluEval ====

## inference for one task with instruction-version llm
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6018  examples/inference/preference_test.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
# --peft_model /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/llama2-peft-2epoch \
# --inference_dataset halueval_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/preference_tuning/fsdp_peft_flash_1500k/llama2-peft-2epoch/halueval_hallucination_predictions


## inference for one task with preference-version llm
torchrun --nnodes 1 --nproc_per_node 1 --master_port 6035  examples/inference/preference_test.py \
--model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
--peft_model /home/jiw203/wjn/InstructGraph/output/preference_tuning/fsdp_peft_flash_1500k/llama2-peft \
--inference_dataset halueval_dataset \
--inference_save_dir /home/jiw203/wjn/InstructGraph/output/preference_tuning/fsdp_peft_flash_1500k/llama2-peft-preference-2epoch/halueval_hallucination_predictions



## inference with only llama2 / vicuna
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6015  examples/inference/preference_test.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
# --inference_dataset halueval_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/preference_tuning/llama2/halueval_hallucination_predictions
