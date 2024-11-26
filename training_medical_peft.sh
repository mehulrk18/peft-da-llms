#!/bin/bash
#echo "Starting training Medical Pubmed Lokr" && \
#time python training_one_v_one.py \
#  --provider "hf" \
#  --peft "lokr" \
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
#echo "Finished training Medical Pubmed Lokr"

#echo "Starting training Medical Pubmed AdaLora" && \
#time python training_one_v_one.py \
#  --provider "hf" \
#  --peft "adalora" \
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
#echo "Finished training Medical Pubmed adalora"

echo "Starting training Medical Pubmed llamaadapter" && \
time python training_one_v_one.py \
  --provider "hf" \
  --peft "llamaadapter" \
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
echo "Finished training Medical Pubmed llamaadapter"

#echo "Starting training Medical Pubmed oft" && \
#time python training_one_v_one.py \
#  --provider "hf" \
#  --peft "oft" \
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
#echo "Finished training Medical Pubmed oft"

echo "!!Trained Medical's Pubmed Datasets on LoRa, Loha, IA3, Lokr, AdaLora, LlamaAdapter, and OFT!!"

#echo "Starting training Medical Scilay ia3" && \
#time python training_one_v_one.py \
#  --provider "hf" \
#  --peft "ia3" \
#  --domain "medical" \
#  --dataset "scilay" \
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
#echo "Finished training Medical Scilay ia3"
#
#echo "Starting training Medical MSLR ia3" && \
#time python training_one_v_one.py \
#  --provider "hf" \
#  --peft "ia3" \
#  --domain "medical" \
#  --dataset "mslr" \
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
#echo "Finished training Medical MSLR ia3"
#
#echo "Starting training Medical Cord19 ia3" && \
#time python training_one_v_one.py \
#  --provider "hf" \
#  --peft "ia3" \
#  --domain "medical" \
#  --dataset "cord19" \
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
#echo "Finished training Medical Cord19 ia3"
#
#echo "!!Trained Medical Datasets on LoRa, Loha and IA3!!"