#!/bin/bash
# 基于 pretrain_1536_resume_step19000.pth 继续预训练（RTX 4080 SUPER 32GB 单卡）
cd /root/autodl-tmp/minimind/trainer

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 TORCH_LOGS=recompiles \
SWANLAB_API_KEY=nNfaXPyhPPvmv2G1OOT45 \
python train_pretrain.py \
  --log_interval 20 \
  --hidden_size 1536 --num_hidden_layers 32 \
  --max_seq_len 2048 --batch_size 16 --accumulation_steps 8 \
  --epochs 1 --learning_rate 5e-4 --lr_scheduler wsd \
  --warmup_ratio 0.01 --decay_ratio 0.2 \
  --gradient_checkpointing 1 --use_compile 1 \
  --data_path /root/autodl-tmp/Nemotron-CC-Math-v1/4plus \
  --tokenizer_path ../model/tokenizer_en \
  --from_resume 1 --resume_dir /root/checkpoints_1b \
  --save_dir /root/minimind-out \
  --num_threads 16 \
  --use_wandb --logger swanlab \
  >> /root/minimind-out/train.log 2>&1
