#!/bin/bash
echo "Starting training Legal EurLex loha" && \
time python training_one_v_one.py \
  --provider "hf" \
  --peft "loha" \
  --domain "legal" \
  --dataset "eurlex" \
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
echo "Finished training Legal EurLex loha"

echo "Starting training Legal BillSum loha" && \
time python training_one_v_one.py \
  --provider "hf" \
  --peft "loha" \
  --domain "legal" \
  --dataset "billsum" \
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
echo "Finished training Legal BillSum loha"

#echo "Starting training Legal MultiLex loha" && \
#time python training_one_v_one.py \
#  --provider "hf" \
#  --peft "loha" \
#  --domain "legal" \
#  --dataset "multilex" \
#  --train_epochs 5 \
#  --tokenization_with_attention "True" \
#  --max_seq_len 4096 \
#  --training_samples 1000 \
#  --eval_samples 250 \
#  --test_samples 1 \
#  --sorted_dataset "True" \
#  --batch_size 4 \
#  --torch_dtype "bf16" \
#  --return_overflowing_tokens "True"
#echo "Finished training Legal MultiLex loha"
