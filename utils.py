#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/6 19:40
# @Author  : LiXin
# @File    : utils.py
# @Describe:
import os
import unicodedata
import jieba
from simhash import Simhash
from string import digits
import re
import math
import time
from gensim.models import word2vec
import jieba
import queue
from jieba import lcut
from gensim.similarities import SparseMatrixSimilarity
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import Levenshtein
from MathUtil import getSim_HFS,getSim_DHFS,getNDCG
from bs4 import BeautifulSoup
def is_chinese(uchar):
    if uchar >= '\u4e00' and uchar <= '\u9fa5':
        return True
    else:
        return False


def reserve_chinese(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str += i
    return content_str

def remove_digits(content):
    return re.sub(r'[0-9]+', '', content)


def getContentSimscore(queryConent,content):
    words1 = jieba.lcut(queryConent, cut_all=True)
    words2 = jieba.lcut(content, cut_all=True)
    dis= Simhash(words1).distance(Simhash(words2))
    u=0.07
    v=0.5
    return v*math.cos(u*dis)+v


def getFilenameID(filename):
    partFilename = filename.split('/')[-1]
    partFilename = os.path.splitext(partFilename)[0]
    temp = str(unicodedata.normalize('NFKD', partFilename).encode('ascii', 'ignore'))
    temp = temp[2:]
    file_name = temp[:-1]
    return file_name

def MemExprByNum(q,d):
    if len(q)==0 or len(d)==0:return 0
    maxs=0
    if len(q)>len(d):
        for i in d:
            if i in q:maxs+=1
    else:
        for i in q:
            if i in d:maxs+=1
    try:
        s=-1/math.exp(maxs)+1
    except Exception as e:
        print(e)
        print(time.asctime(time.localtime(time.time())) + str(maxs))
        return 0
    return s
def MemExprByNum1(q,d):
    if len(q)==0 or len(d)==0:return 0
    maxs=0
    if len(q)>len(d):
        for i in d:
            if i in q:maxs+=1
    else:
        for i in q:
            if i in d:maxs+=1
    try:
        s=-1/math.exp(maxs)+1
    except Exception as e:
        print(e)
        print(time.asctime(time.localtime(time.time())) + str(maxs))
        return 0
    return s

def MemExprByDHFS(q,d):
    if len(q)==0 or len(d)==0:return 0
    maxs=0
    for i in q:
        for j in d:
            try:
                a=eval(dict(i)['FormularFDS'])
                b=eval(j)
                sim=getSim_DHFS(a,b)
                if sim>maxs:maxs=sim
            except:
                print(time.asctime(time.localtime(time.time())) + "比较公式相似度中，公式符号存在'\\' ")
    return maxs

def MemRef(q,d):
    if len(q) == 0 or len(d) == 0: return 0
    l=len(q)
    sc=0
    maxs=0
    for i in q:
        if len(q)==0:return 0
        # 1、将【文本集】生成【分词列表】
        texts = [lcut(text) for text in d]
        # 2、基于文本集建立【词典】，并获得词典特征数
        dictionary = Dictionary(texts)
        num_features = len(dictionary.token2id)
        # 3.1、基于词典，将【分词列表集】转换成【稀疏向量集】，称作【语料库】
        corpus = [dictionary.doc2bow(text) for text in texts]
        # 3.2、同理，用【词典】把【搜索词】也转换为【稀疏向量】
        kw_vector = dictionary.doc2bow(lcut(i))
        # 4、创建【TF-IDF模型】，传入【语料库】来训练
        tfidf = TfidfModel(corpus)
        # 5、用训练好的【TF-IDF模型】处理【被检索文本】和【搜索词】
        tf_texts = tfidf[corpus]  # 此处将【语料库】用作【被检索文本】
        tf_kw = tfidf[kw_vector]
        # 6、相似度计算
        sparse_matrix = SparseMatrixSimilarity(tf_texts, num_features)
        similarities = sparse_matrix.get_similarities(tf_kw)
        for e, s in enumerate(similarities, 1):
            sc+=s
            # print('kw 与 text%d 相似度为：%.2f' % (e, s))
            if(s>maxs):maxs=s
    return maxs

def MemKey(q,d):
    if len(q) == 0 or len(d) == 0: return 0
    """
    keyword计算隶属度
    :param q:
    :param d:
    :return:
    """
    return 1 if len(list(set(q) & set(d)))!=0 else 0


def MemTit(q,d):
    if q is None or d is None: return 0
    if len(q)==0 or len(d)==0:return 0
    """
    title计算隶属度
    :param q:
    :param d:
    :return:
    """
    dis = Levenshtein.distance(q, d)
    score = 1-dis/max(len(q),len(d))
    a=math.sin(math.pi/2*score)
    return a


def MemERr(q,d):
    if len(q)==0 and len(d)==0:return 1
    if len(q) == 0 or len(d) == 0: return 0
    return min(len(q),len(d))/len(q)


def getContentSimByTwoFile(file1,file2):
    queryConetent = reserve_chinese(BeautifulSoup(open(file1, encoding="utf-8", errors='ignore')).find('div', class_='content').text)
    conetent = reserve_chinese(BeautifulSoup(open(file2, encoding="utf-8", errors='ignore')).find('div', class_='content').text)
    contentSim = getContentSimscore(queryConetent, conetent)
    return contentSim

if __name__ == '__main__':
    file="中文数据集/J计算机应用与软件/J计算机应用与软件201936-07-300-306.html"
    file1 = "中文数据集/J计算机工程/J计算机工程201945-09-242-247.html"
    print(getContentSimByTwoFile(file,file1))
    file2="中文数据集/J计算机应用/J计算机应用201939-11-3198-3203.html"
    print(getContentSimByTwoFile(file, file2))
    file3="中文数据集/J计算机应用/J计算机应用201939-03-700-705.html"
    print(getContentSimByTwoFile(file, file3))
    file4="中文数据集/J计算机应用/J计算机应用201939-04-1012-1020.html"
    print(getContentSimByTwoFile(file, file4))
    file5="中文数据集/J计算机应用/J计算机应用201939-03-924-929.html"
    print(getContentSimByTwoFile(file, file5))
