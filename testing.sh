#!/bin/bash
  time python testing_one_v_one_model.py \
  --trained_peft_path saved_models/medical_oft_hf_07-11-2024_23-29-12/medical_oft \
  --training_samples 1 \
  --eval_samples 1 \
  --test_samples 2 \
  --sorted_dataset "True"
#  --trained_peft_path saved_models/hf_medical_loha_1000_5_4_bs_27-10-2024_07-16-42_summarization/medical_loha \
#  --trained_peft_path results/llama_simple_medical_lora_ah_2000_30_bs_4096_4_04-11-2024_15-49-14/checkpoint-500/medical_lora \
#  --chat_template "True"
