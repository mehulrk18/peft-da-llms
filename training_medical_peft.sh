#!/bin/bash
#echo "Starting training Medical Pubmed Lora" && \
#time python training_one_v_one.py \
#  --provider "hf" \
#  --peft "lora" \
#  --domain "medical" \
#  --dataset "pubmed" \
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
#echo "Finished training Medical Pubmed Lora"


echo "Starting training Medical Scilay Lora" && \
time python training_one_v_one.py \
  --provider "hf" \
  --peft "lora" \
  --domain "medical" \
  --dataset "scilay" \
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
echo "Finished training Medical Scilay Lora"

echo "Starting training Medical MSLR Lora" && \
time python training_one_v_one.py \
  --provider "hf" \
  --peft "lora" \
  --domain "medical" \
  --dataset "mslr" \
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
echo "Finished training Medical MSLR Lora"

echo "Starting training Medical Cord19 Lora" && \
time python training_one_v_one.py \
  --provider "hf" \
  --peft "lora" \
  --domain "medical" \
  --dataset "cord19" \
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
echo "Finished training Medical Cord19 Lora"