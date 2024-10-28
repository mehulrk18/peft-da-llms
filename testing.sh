#!/bin/bash
  time python testing_one_v_one_model.py \
  --training_samples 1 \
  --eval_samples 1 \
  --test_samples 250 \
  --trained_peft_path saved_models/hf_medical_loha_1000_5_4_bs_27-10-2024_07-16-42_summarization/medical_loha \
  --sorted_dataset "True"
#  --chat_template "True"
#  --trained_peft_path results/llama_simple_medical_lora_ah_1000_2_bs_2048_4_21-10-2024_12-06-22/checkpoint-256/medical_lora \
