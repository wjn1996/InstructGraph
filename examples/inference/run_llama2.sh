export CUDA_VISIBLE_DEVICES=7
## inference for all tasks
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6015  examples/inference/llama.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
# --peft_model /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/llama2-peft \
# --inference_dataset instructgraph_dataset \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/llama2-peft/predictions

## inference for one task
torchrun --nnodes 1 --nproc_per_node 1 --master_port 6054  examples/inference/llama.py \
--model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
--peft_model /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/llama2-peft-2epoch \
--inference_dataset instructgraph_dataset \
--inference_task graph-construction-modeling-structure-graph-generation-directedweighted \
--inference_save_dir /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/llama2-peft-2epoch/predictions

## inference with only llama2 / vicuna
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6019  examples/inference/llama.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
# --inference_dataset instructgraph_dataset \
# --inference_task graph-construction-modeling-structure-graph-generation-directedweighted \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/instruction_tuning/llama2/predictions
