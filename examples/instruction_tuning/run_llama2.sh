export CUDA_VISIBLE_DEVICES=0,3,4,5,6,7
torchrun --nnodes 1 --nproc_per_node 6 --master_port 6013 examples/instruction_tuning/llama.py \
--enable_fsdp \
--use_peft \
--context_length 2048 \
--batching_strategy padding \
--peft_method lora \
--model_name /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
--fsdp_config.pure_bf16 \
--output_dir output/instruction_tuning/fsdp_peft_1500k/llama2-peft \
