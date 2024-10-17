#!/bin/bash
  python testing_one_v_one_model.py \
  --trained_peft_path results/llama_simple_hf_medical_ia3_1000_10_2048_bs_4_16-10-2024_02-34-59/checkpoint-2560/medical_ia3 \
  --training_samples 1 \
  --eval_samples 1 \
  --test_samples 20 \
  --sorted_dataset "True"
#  --chat_template "True" \
