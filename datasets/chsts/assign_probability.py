#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: assign_probability.py
# @time: 2020-11-19 22:49
# @desc:

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

tokenized_text = tokenizer.tokenize('我爱你')

print(tokenized_text)