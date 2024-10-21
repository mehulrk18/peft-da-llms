#!/bin/bash
echo "Starting training" && \
python training_one_v_one.py \
  --peft "lora" \
  --domain "medical" \
  --provider "hf" \
  --train_epochs 1 \
  --tokenization_with_attention "True" \
  --max_seq_len 2048 \
  --training_samples 4 \
  --eval_samples 4 \
  --test_samples 1 \
  --sorted_dataset "True" \
  --batch_size 4 \
  --return_overflowing_tokens "True"
#  --chat_template "True" \
#  --use_instruct_model "True" \
#  --mlm "True" \
