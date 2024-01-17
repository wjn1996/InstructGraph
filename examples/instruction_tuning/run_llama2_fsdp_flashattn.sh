export CUDA_VISIBLE_DEVICES=0,3,4,5,6,7
torchrun --nnodes 1 --nproc_per_node 6 --master_port 6015  examples/instruction_tuning/llama.py \
--enable_fsdp \
--context_length 2048 \
--batching_strategy padding \
--model_name /home/jiw203/wjn/pre-trained-lm/Llama-2-7b-hf \
--dist_checkpoint_root_folder /home/jiw203/wjn/InstructGraph/output/instruction_tuning/fsdp_flash_150k \
--dist_checkpoint_folder llama2-fsdp \
--use_fast_kernels