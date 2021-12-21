#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/6 17:36
# @Author  : LiXin
# @File    : contentExtr.py
# @Describe:
import os

def extr():
    dir = "中文数据集"
    for fileDir in os.listdir(dir):

        dirChild = dir + "/" + fileDir
        for file in os.listdir(dirChild):
            filename = dirChild + "/" + file
            print("正在创建..." + filename)
            # extrSingleFile(filename)


# def extrSingleFile(filename):