#!/usr/bin/env bash
python train.py --base_model bert-base-uncased --sentence_embedding_dim 256
--model_save_path ./model_save --batch_size 16 --epochs 10 --dataset msrp
--task_type classification


