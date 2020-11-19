#!/usr/bin/env bash
python train.py --base_model bert-base-uncased --sentence_embedding_dim 256
--model_save_path ./model_save --batch_size 16 --epochs 10 --dataset msrp
--task_type classification

CUDA_VISIBLE_DEVICES=1 python train.py --base_model sentence-transformers/distilbert-base-nli-stsb-mean-tokens --sentence_embedding_dim 256 --model_save_path ./atec_model_save --batch_size 64 --epochs 5 --dataset atec --task_type classification

CUDA_VISIBLE_DEVICES=1 python eval.py --trained_model_path XXXXX --output_dir ./XXperformance/ --task_type classification --dataset atec