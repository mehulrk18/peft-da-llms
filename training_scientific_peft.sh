#!/bin/bash
#echo "Starting training Scientific Arxiv Lokr" && \
#time python training_one_v_one.py \
#  --provider "hf" \
#  --peft "lokr" \
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
#echo "Finished training Scientific Arxiv Lokr"

#echo "Starting training Scientific Arxiv AdaLora" && \
#time python training_one_v_one.py \
#  --provider "hf" \
#  --peft "adalora" \
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
#echo "Finished training Scientific Arxiv adalora"

echo "Starting training Scientific Arxiv llamaadapter" && \
time python training_one_v_one.py \
  --provider "hf" \
  --peft "llamaadapter" \
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
echo "Finished training Scientific Arxiv llamaadapter"

#echo "Starting training Scientific Arxiv oft" && \
#time python training_one_v_one.py \
#  --provider "hf" \
#  --peft "oft" \
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
#echo "Finished training Scientific Arxiv oft"

echo "!!Trained Scientific's Arxiv Datasets on LoRa, Loha, IA3, Lokr, AdaLora, LlamaAdapter, and OFT!!"



#echo "Starting training Scientific Elsevier ia3" && \
#time python training_one_v_one.py \
#  --provider "hf" \
#  --peft "ia3" \
#  --domain "scientific" \
#  --dataset "elsevier" \
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
#echo "Finished training Scientific Elsevier ia3"
#
#echo "Starting training Scientific Scitldr ia3" && \
#time python training_one_v_one.py \
#  --provider "hf" \
#  --peft "ia3" \
#  --domain "scientific" \
#  --dataset "scitldr" \
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
#echo "Finished training Scientific Scitldr ia3"
#
#echo "!!Trained Scientific Datasets on LoRa, Loha and IA3!!"