#!/usr/bin/env bash

pip install --editable .
python examples/run_tmp_seq.py \
  --task_name TEMPORALVERB \
  --do_train \
  --do_lower_case \
  --data_dir data/tmp_seq_data \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./models/taco_lm
