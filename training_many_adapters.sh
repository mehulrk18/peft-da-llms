#!/bin/bash
echo "Starting training News CNNDM Loha" && \
time python training_one_v_one.py \
  --provider "hf" \
  --peft "loha" \
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
echo "Finished training Medical PubMed Loha"


echo "Starting training Medical PubMed Loha" && \
time python training_one_v_one.py \
  --provider "hf" \
  --peft "loha" \
  --domain "medical" \
  --dataset "pubmed" \
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
echo "Finished training Medical PubMed Loha"

echo "Starting training Scientific Arxiv Loha" && \
time python training_one_v_one.py \
  --provider "hf" \
  --peft "loha" \
  --domain "scientific" \
  --dataset "arxiv" \
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
echo "Finished training Scientific Arxiv Loha"

echo "Starting training Legal Multi_lex Loha" && \
time python training_one_v_one.py \
  --provider "hf" \
  --peft "loha" \
  --domain "legal" \
  --dataset "multi_lex" \
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
echo "Finished training Legal Multi_lex Loha"