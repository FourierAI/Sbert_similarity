#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: load_data.py
# @time: 2020-11-18 16:22
# @desc:
from sentence_transformers import InputExample


def load_dataset(dataset_name, dataset_type):
    if dataset_name == 'atec':
        if dataset_type == "train":
            file_path = 'datasets/atec/train.csv'
        else:
            file_path = 'datasets/atec/test.csv'
    elif dataset_name == 'ccks':
        if dataset_type == "train":
            file_path = 'datasets/ccks/train.txt'
        else:
            file_path = 'datasets/ccks/test.txt'
    elif dataset_name == 'chsts':
        if dataset_type == 'train':
            file_path = 'datasets/chsts/train.txt'
        else:
            file_path = 'datasets/chsts/test.txt'
    elif dataset_name == 'msrp':
        if dataset_type == 'train':
            file_path = 'datasets/msrp/train.txt'
        else:
            file_path = 'datasets/msrp/test.txt'
    else:
        if dataset_type == 'train':
            file_path = 'datasets/sts/train.txt'
        else:
            file_path = 'datasets/sts/test.txt'

    examples = []
    with open(file_path) as file:
        for line in file:
            content = line.split('\t')
            sent1 = content[0]
            sent2 = content[1]
            score = float(content[2].strip())
            example = InputExample(texts=[sent1, sent2], label=score)
            examples.append(example)

    return examples
