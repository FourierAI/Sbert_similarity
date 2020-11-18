#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: clean.py
# @time: 2020-11-18 16:49
# @desc:
from itertools import islice

train_data = []
with open('raw/msr_paraphrase_train.txt') as file:
    for line in islice(file, 1, None):
        content = line.split('\t')
        sent1 = content[3]
        sent2 = content[4].strip()
        label = content[0] + '\n'
        new_line = '\t'.join([sent1, sent2, label])
        train_data.append(new_line)

with open('train.txt', 'a+') as file:
    for line in train_data:
        file.write(line)

test_data = []
with open('raw/msr_paraphrase_test.txt') as file:
    for line in islice(file, 1, None):
        content = line.split('\t')
        sent1 = content[3]
        sent2 = content[4].strip()
        label = content[0] + '\n'
        new_line = '\t'.join([sent1, sent2, label])
        test_data.append(new_line)

with open('test.txt', 'a+') as file:
    for line in test_data:
        file.write(line)
