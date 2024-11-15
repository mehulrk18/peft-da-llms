#!/bin/bash
echo "Starting training" && \
time python training_one_v_one.py \
  --provider "hf" \
  --peft "lora" \
  --domain "news" \
  --dataset "cnn_dm" \
  --train_epochs 5 \
  --tokenization_with_attention "True" \
  --max_seq_len 4096 \
  --training_samples 1000 \
  --eval_samples 250 \
  --test_samples 1 \
  --sorted_dataset "True" \
  --batch_size 4 \
  --torch_dtype "bf16" \
  --return_overflowing_tokens "True"
#  --do_inference "True" \
#  --chat_template "True" \
#  --use_instruct_model "True" \
#  --mlm "True" \
