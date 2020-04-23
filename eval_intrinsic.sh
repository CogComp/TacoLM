#!/usr/bin/env bash

pip install --editable .
python examples/run_tmp_seq_eval.py \
  --task_name TEMPORALVERB \
  --do_eval \
  --do_lower_case \
  --data_dir data/intrinsic/bert.realnews.txt \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir eval_results/dev_null

pip install --editable .
python examples/run_tmp_seq_eval.py \
  --task_name TEMPORALVERB \
  --do_eval \
  --do_lower_case \
  --data_dir data/intrinsic/bert.udstmp.txt \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir eval_results/dev_null

pip install --editable .
python examples/run_tmp_seq_eval.py \
  --task_name TEMPORALVERB \
  --do_eval \
  --do_lower_case \
  --data_dir data/intrinsic/model.realnews.txt \
  --bert_model models/taco_lm_epoch_3  \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir eval_results/dev_null

pip install --editable .
python examples/run_tmp_seq_eval.py \
  --task_name TEMPORALVERB \
  --do_eval \
  --do_lower_case \
  --data_dir data/intrinsic/model.udstmp.txt \
  --bert_model models/taco_lm_epoch_3 \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir eval_results/dev_null

