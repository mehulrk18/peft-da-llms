#!/bin/bash
#python testing_one_v_one_model.py --checkpoint results/llama_medical_lora_hf_1000_2_2048_06-10-2024_16-57-35/checkpoint-500 \
#  --trained_peft_path saved_models/hf_medical_lora_1000_2_06-10-2024_16-57-35_summarization_method2 \
  python testing_one_v_one_model.py --trained_peft_path results/llama_medical_lora_hf_1000_2_2048_06-10-2024_16-57-35/checkpoint-500/medical_lora \
  --test_samples 10
