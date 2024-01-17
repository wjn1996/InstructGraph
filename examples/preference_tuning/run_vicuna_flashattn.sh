export CUDA_VISIBLE_DEVICES=4,5,6,7

# pre-trained peft model in instruction-tuning stage
PRETRAINED_PEFT=/home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_peft_flash_1500k/vicuna-peft-2epoch

torchrun --nnodes 1 --nproc_per_node 4 --master_port 6023 examples/preference_tuning/llama.py \
--enable_fsdp \
--use_peft \
--context_length 2048 \
--batching_strategy padding \
--peft_method lora \
--model_name /home/jiw203/wjn/pre-trained-lm/vicuna-7b-v1.5 \
--peft_model $PRETRAINED_PEFT \
--fsdp_config.pure_bf16 \
--output_dir output/preference_tuning/fsdp_peft_flash_1500k/vicuna-peft \
--use_fast_kernels