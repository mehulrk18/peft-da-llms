#!/bin/bash
  python testing_one_v_one_model.py \
  --trained_peft_path results/llama_instruct_mlm_hf_scientific_lora_1000_2_8192_bs_2_11-10-2024_02-22-08/checkpoint-500/scientific_lora \
  --training_samples 1 \
  --eval_samples 1 \
  --test_samples 5 \
  --sorted_dataset "True" \
  --chat_template "True"