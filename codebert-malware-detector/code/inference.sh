#!/bin/bash

echo "Starting Testing with Finetuned CodeBERT..."
echo "Start Time: $(date)"
echo "Logging to: ../logs/test.log"
echo "--------------------------------------------"

python3 run.py \
  --model_type roberta \
  --tokenizer_name microsoft/codebert-base \
  --model_name_or_path ../saved_models/codebert-finetuned/checkpoint-best-acc \
  --do_eval \
  --do_test \
  --train_data_file ../data/train.jsonl \
  --eval_data_file ../data/val.jsonl \
  --test_data_file ../data/test.jsonl \
  --output_dir ../saved_models/codebert-finetuned \
  --block_size 400 \
  --train_batch_size 32 \
  --eval_batch_size 64 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --epoch 5 \
  --evaluate_during_training \
  --seed 123456 2>&1 | tee ../logs/test.log

echo "Inference finished at: $(date)"