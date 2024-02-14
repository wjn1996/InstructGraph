export CUDA_VISIBLE_DEVICES=7
## inference for all tasks
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6015  examples/inference/llama.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/vicuna-7b-v1.5 \
# --peft_model /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/vicuna-peft-2epoch \
# --inference_dataset instructgraph_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/vicuna-peft-2epoch/predictions

## inference for one task
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6017 examples/inference/llama.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/vicuna-7b-v1.5 \
# --peft_model /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/vicuna-peft-1epoch \
# --inference_dataset instructgraph_dataset \
# --inference_task graph-construction-modeling-structure-graph-generation-directedweighted \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/vicuna-peft-1epoch/predictions

## inference with only llama2 / vicuna
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6028  examples/inference/llama.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/vicuna-7b-v1.5 \
# --inference_dataset instructgraph_dataset \
# --inference_task graph-language-modeling-graph-question-answering-pathquestion \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/instruction_tuning/vicuna/predictions



# ==== BBH Benchmark ====

## inference for one task with instruction-version llm
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6021  examples/inference/llama.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/vicuna-7b-v1.5 \
# --peft_model /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/vicuna-peft-2epoch \
# --inference_dataset big_bench_hard_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/vicuna-peft-2epoch/bbh_predictions

## inference with only llama2 / vicuna
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6031  examples/inference/llama.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/vicuna-7b-v1.5 \
# --inference_dataset big_bench_hard_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/instruction_tuning/vicuna/bbh_predictions

# ==== MMLU Benchmark ====

# inference for one task with instruction-version llm
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6022  examples/inference/llama.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/vicuna-7b-v1.5 \
# --peft_model /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/vicuna-peft-2epoch \
# --inference_dataset mmlu_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/vicuna-peft-2epoch/mmlu_predictions

## inference with only llama2 / vicuna
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6030  examples/inference/llama.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/vicuna-7b-v1.5 \
# --inference_dataset mmlu_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/instruction_tuning/vicuna/mmlu_predictions


# ==== Planing Benchmark ====
## inference with only llama2 / vicuna
torchrun --nnodes 1 --nproc_per_node 1 --master_port 6063  examples/inference/llama.py \
--model_name_or_path /home/jiw203/wjn/pre-trained-lm/vicuna-7b-v1.5 \
--inference_dataset planning_dataset \
--inference_save_dir /home/jiw203/wjn/InstructGraph/output/instruction_tuning/vicuna/planning_predictions