#!/bin/bash
echo "Starting training News MultiNews Loha" && \
time python training_one_v_one.py \
  --provider "hf" \
  --peft "loha" \
  --domain "news" \
  --dataset "multinews" \
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
echo "Finished training News MultiNews Loha"


echo "Starting training News XSum Loha" && \
time python training_one_v_one.py \
  --provider "hf" \
  --peft "loha" \
  --domain "news" \
  --dataset "xsum" \
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
echo "Finished training News XSum Loha"

echo "Starting training News Newsroom Loha" && \
time python training_one_v_one.py \
  --provider "hf" \
  --peft "loha" \
  --domain "news" \
  --dataset "newsroom" \
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
echo "Finished training News Newsroom Loha"

#echo "Starting training News CNNDM Loha" && \
#time python training_one_v_one.py \
#  --provider "hf" \
#  --peft "loha" \
#  --domain "legal" \
#  --dataset "cnndm" \
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
#echo "Finished training Legal CNNDM Loha"