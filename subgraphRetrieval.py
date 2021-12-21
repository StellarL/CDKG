#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/22 16:57
# @Author  : LiXin
# @File    : subgraphRetrieval.py
# @Describe:

from formularRetrieval import getSubGraph

if __name__ == '__main__':


    r=getSubGraph("J201941-11-1939-1948:5")
    score = sorted(r.items(), key=lambda x: x[1], reverse=True)
    print(score)
    # get_ndcg("J201936-07-300-306:63",r)