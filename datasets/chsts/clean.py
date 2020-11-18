#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: clean.py
# @time: 2020-11-18 16:49
# @desc:
import random

ch_sts = []
with open('raw/simtrain_to05sts.txt') as file:
    for line in file:
        content = line.split('\t')
        sent1 = content[1]
        sent2 = content[3]
        label = content[4]
        line_content = '\t'.join([sent1, sent2, label])
        ch_sts.append(line_content)

with open('raw/simtrain_to05sts_same.txt') as file:
    for line in file:
        content = line.split('\t')
        sent1 = content[1]
        sent2 = content[3]
        label = content[4]
        line_content = '\t'.join([sent1, sent2, label])
        ch_sts.append(line_content)

random.shuffle(ch_sts)

ch_sts_len = len(ch_sts)
train_ch_sts = ch_sts[:ch_sts_len // 5 * 4]
test_ch_sts = ch_sts[ch_sts_len // 5 * 4:]

with open('train.txt', 'a+') as file:
    for line in train_ch_sts:
        file.write(line)

with open('test.txt', 'a+') as file:
    for line in test_ch_sts:
        file.write(line)
