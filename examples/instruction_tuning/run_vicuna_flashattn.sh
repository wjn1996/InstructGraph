export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nnodes 1 --nproc_per_node 4 --master_port 6013 examples/instruction_tuning/llama.py \
--enable_fsdp \
--use_peft \
--context_length 2048 \
--batching_strategy padding \
--peft_method lora \
--model_name /home/jiw203/wjn/pre-trained-lm/vicuna-7b-v1.5 \
--fsdp_config.pure_bf16 \
--output_dir output/instruction_tuning/fsdp_peft_flash_1500k/vicuna-peft \
--use_fast_kernels