#!/bin/bash
  python testing_one_v_one_model.py \
  --training_samples 1 \
  --eval_samples 1 \
  --test_samples 250 \
  --trained_peft_path saved_models/hf_medical_llama_adapter_1000_5_4_bs_25-10-2024_19-06-06_summarization/medical_llama_adapter \
  --sorted_dataset "True"
#  --chat_template "True"
#  --trained_peft_path results/llama_simple_medical_lora_ah_1000_2_bs_2048_4_21-10-2024_12-06-22/checkpoint-256/medical_lora \
