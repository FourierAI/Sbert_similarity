import argparse
import json
from itertools import islice

from sklearn import metrics
from scipy.stats import spearmanr
from scipy.stats import pearsonr

from sentence_transformers import SentenceTransformer, util

import load_data as ld


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_model_path', type=str, default='./')
    parser.add_argument('--output_file', type=str, default='performance.json')
    parser.add_argument('--dataset', type=str, default='msrp', choices=['msrp', 'sts', 'chinese_bank'])
    parser.add_argument('--task_type', type=str, default='', choices=['classification', 'regression'])
    args = parser.parse_args()

    trained_model_path = args.trained_model_path
    output_file = args.output_file
    dataset = args.dataset
    task_type = args.task_type

    test_examples = ld.load_dataset(dataset_name=dataset, dataset_type='test')

