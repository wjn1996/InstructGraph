export CUDA_VISIBLE_DEVICES=2
SHOT=2

## inference for one task
# torchrun --nnodes 1 --nproc_per_node 1 --master_port 6022  examples/inference/llama.py \
# --model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
# --peft_model /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/llama2-peft-1.5epoch \
# --reasoning_type few-shot-icl \
# --shot $SHOT \
# --inference_dataset instructgraph_dataset \
# --inference_task graph-language-modeling-graph-question-answering-wikitablequestions \
# --inference_save_dir /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/llama2-peft-1.5epoch/predictions_$SHOT-shot_icl

## inference with only llama2 / vicuna
torchrun --nnodes 1 --nproc_per_node 1 --master_port 6017 examples/inference/llama.py \
--model_name_or_path /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
--reasoning_type few-shot-icl \
--shot $SHOT \
--inference_dataset instructgraph_dataset \
--inference_task graph-construction-modeling-structure-graph-generation-directedweighted \
--inference_save_dir /home/jiw203/wjn/InstructGraph/output/instruction_tuning/llama2/predictions_$SHOT-shot_icl
