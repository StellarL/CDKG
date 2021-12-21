#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/30 19:22
# @Author  : LiXin
# @File    : data.py
# @Describe:
from transformers import *
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

