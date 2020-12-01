#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: assign_probability.py
# @time: 2020-11-19 22:49
# @desc:

from transformers import AutoTokenizer
import random


def generate_probability():
    while True:
        random_value = random.uniform(0, 1)
        yield random_value


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

words = set()
with open('test.txt') as file:
    for line in file:
        lines = line.split('\t')
        sent1 = lines[0]
        sent2 = lines[1]
        words1 = tokenizer.tokenize(sent1)
        words2 = tokenizer.tokenize(sent2)
        words.update(words1)
        words.update(words2)

with open('train.txt') as file:
    for line in file:
        lines = line.split('\t')
        sent1 = lines[0]
        sent2 = lines[1]
        words1 = tokenizer.tokenize(sent1)
        words2 = tokenizer.tokenize(sent2)
        words.update(words1)
        words.update(words2)

with open('words.txt', 'a+') as file:
    for word in words:
        weight = next(generate_probability())
        file.write(word + '\t' + str(weight) + '\n')
