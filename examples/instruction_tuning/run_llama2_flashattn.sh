export CUDA_VISIBLE_DEVICES=1,3,4,7
torchrun --nnodes 1 --nproc_per_node 4 --master_port 6015 examples/instruction_tuning/llama.py \
--enable_fsdp \
--use_peft \
--batch_size_training 4 \
--context_length 2048 \
--batching_strategy padding \
--peft_method lora \
--model_name /home/jiw203/wjn/pre-trained-lm/Llama-2-13b-hf \
--fsdp_config.pure_bf16 \
--output_dir output/instruction_tuning/fsdp_peft_flash_1500k/llama2-13b-peft \
--use_fast_kernels

# Llama-2-7b-hf: context_length=2048, nproc_per_node=4, batch_size_training=8, gradient_accumulation_steps=1, epoch=2
# Llama-2-13b-hf: context_length=2048, nproc_per_node=4, batch_size_training=4, gradient_accumulation_steps=1, epoch=2