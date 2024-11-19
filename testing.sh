#!/bin/bash
echo "Testing on Scientific Arxiv Lora for 50 Samples"
time python testing_one_v_one_model.py \
  --trained_peft_path trained_pefts/hf_scientific_arxiv_lora_checkpoint-1228/scientific_arxiv_lora \
  --training_samples 1 \
  --eval_samples 1 \
  --test_samples 50 \
  --sorted_dataset "True"
echo "Testing on Scientific Arxiv Lora Finished"

echo "Testing on Medical Pubmed Lora for 50 Samples"
time python testing_one_v_one_model.py \
  --trained_peft_path trained_pefts/hf_medical_pubmed_lora_checkpoint-500/medical_pubmed_lora \
  --training_samples 1 \
  --eval_samples 1 \
  --test_samples 50 \
  --sorted_dataset "True"
echo "Testing on Medical Pubmed Lora Finished"

echo "Testing on News CNN DM Lora for 50 Samples"
time python testing_one_v_one_model.py \
  --trained_peft_path trained_pefts/hf_news_cnn_dm_lora_checkpoint-500/news_cnn_dm_lora \
  --training_samples 1 \
  --eval_samples 1 \
  --test_samples 50 \
  --sorted_dataset "True"
echo "Testing on News CNN DM Lora Finished"

echo "Testing on Legal MultiLEx Lora for 50 Samples"
time python testing_one_v_one_model.py \
  --trained_peft_path trained_pefts/hf_legal_multi_lex_lora_checkpoint-2466/legal_multi_lex_lora \
  --training_samples 1 \
  --eval_samples 1 \
  --test_samples 50 \
  --sorted_dataset "True"
echo "Testing on Scientific Arxiv Lora Finished"








#  --trained_peft_path saved_models/hf_medical_loha_1000_5_4_bs_27-10-2024_07-16-42_summarization/medical_loha \
#  --trained_peft_path results/llama_simple_medical_lora_ah_2000_30_bs_4096_4_04-11-2024_15-49-14/checkpoint-500/medical_lora \
#  --chat_template "True"
