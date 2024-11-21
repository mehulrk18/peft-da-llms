#!/bin/bash
#echo "Starting training Scientific Arxiv Loha" && \
#time python training_one_v_one.py \
#  --provider "hf" \
#  --peft "loha" \
#  --domain "scientific" \
#  --dataset "arxiv" \
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
#echo "Finished training Scientific PubMed Loha"


echo "Starting training Scientific Elsevier Loha" && \
time python training_one_v_one.py \
  --provider "hf" \
  --peft "loha" \
  --domain "scientific" \
  --dataset "elsevier" \
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
echo "Finished training Scientific Elsevier Loha"

echo "Starting training Scientific Scitldr Loha" && \
time python training_one_v_one.py \
  --provider "hf" \
  --peft "loha" \
  --domain "scientific" \
  --dataset "scitldr" \
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
echo "Finished training Scientific Scitldr Loha"
