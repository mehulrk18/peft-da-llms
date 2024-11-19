#!/bin/bash
#echo "Starting training Scientific Arxiv Lora" && \
#time python training_one_v_one.py \
#  --provider "hf" \
#  --peft "lora" \
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
#echo "Finished training Scientific PubMed Lora"


echo "Starting training Scientific Elsevier Lora" && \
time python training_one_v_one.py \
  --provider "hf" \
  --peft "lora" \
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
echo "Finished training Scientific Elsevier Lora"

echo "Starting training Scientific Scitldr Lora" && \
time python training_one_v_one.py \
  --provider "hf" \
  --peft "lora" \
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
echo "Finished training Scientific Scitldr Lora"

#echo "Starting training Legal Multi_lex Lora" && \
#time python training_one_v_one.py \
#  --provider "hf" \
#  --peft "lora" \
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
#echo "Finished training Legal Multi_lex Lora"