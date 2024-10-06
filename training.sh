#!/bin/bash
python training_one_v_one.py --peft lora \
  --domain medical \
  --train_epochs 2 \
  --max_seq_len 2048 \
  --tok "True" \
  --training_samples 1000 \
  --mlm "True"
