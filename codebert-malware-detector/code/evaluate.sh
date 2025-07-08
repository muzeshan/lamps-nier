#!/bin/bash

python3 ../evaluator/evaluator.py \
  --answers ../data/test.jsonl \
  --predictions ../saved_models/codebert-finetuned/predictions.txt

