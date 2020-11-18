#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: clean.py
# @time: 2020-11-18 16:49
# @desc:

train_data = []
with open('raw/train.csv') as file:
    for line in file:
        content = line.split('\t')
        sent1 = content[1]
        sent2 = content[2]
        label = content[3]
        new_line = '\t'.join([sent1, sent2, label])
        train_data.append(new_line)

with open('train.csv', 'a+') as file:
    for line in train_data:
        file.write(line)

test_data = []
with open('raw/test.csv') as file:
    for line in file:
        content = line.split('\t')
        sent1 = content[1]
        sent2 = content[2]
        label = content[3]
        new_line = '\t'.join([sent1, sent2, label])
        test_data.append(new_line)

with open('test.csv', 'a+') as file:
    for line in test_data:
        file.write(line)