#!/bin/bash
echo "Starting training" && \
python training_one_v_one.py \
  --peft lora \
  --domain medical \
  --train_epochs 1 \
  --tokenization_with_attention "True" \
  --max_seq_len 4096 \
  --training_samples 1000 \
  --eval_samples 1000 \
  --test_samples 1 \
  --sorted_dataset "True" \
  --use_instruct_model "True" \
  --batch_size 2 \
  --chat_template "True" \
  --return_overflowing_tokens "True"
#  --mlm "True" \
