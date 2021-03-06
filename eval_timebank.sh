#!/usr/bin/env bash

pip install --editable .
python examples/run_hieve.py \
  --task_name TEMPORALVERB \
  --do_train \
  --do_eval \
  --data_dir data/timebank \
  --bert_model models/taco_lm_epoch_2 \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --seed 10 \
  --output_dir eval_results/timebank_1

pip install --editable .
python examples/run_hieve.py \
  --task_name TEMPORALVERB \
  --do_train \
  --do_eval \
  --data_dir data/timebank \
  --bert_model models/taco_lm_epoch_2 \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --seed 20 \
  --output_dir eval_results/timebank_2

pip install --editable .
python examples/run_hieve.py \
  --task_name TEMPORALVERB \
  --do_train \
  --do_eval \
  --data_dir data/timebank \
  --bert_model models/taco_lm_epoch_2 \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --seed 30 \
  --output_dir eval_results/timebank_3

python examples/run_hieve.py \
  --task_name TEMPORALVERB \
  --do_train \
  --do_eval \
  --data_dir data/timebank \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --seed 10 \
  --output_dir eval_results/timebank_bert_1

pip install --editable .
python examples/run_hieve.py \
  --task_name TEMPORALVERB \
  --do_train \
  --do_eval \
  --data_dir data/timebank \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --seed 20 \
  --output_dir eval_results/timebank_bert_2

pip install --editable .
python examples/run_hieve.py \
  --task_name TEMPORALVERB \
  --do_train \
  --do_eval \
  --data_dir data/timebank \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --seed 30 \
  --output_dir eval_results/timebank_bert_3
