#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: load_data.py
# @time: 2020-11-18 16:22
# @desc:

def load_atec_dataset(dataset_type):
    if dataset_type == "train":
        file_path = 'datasets/atec/train.csv'
    else:
        file_path = 'datasets/atec/'