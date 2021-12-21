#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/14 15:50
# @Author  : LiXin
# @File    : formularRetrieval.py
# @Describe:检索
from bs4 import BeautifulSoup
from TangentS.math_tan.math_extractor import MathExtractor
from MathUtil import tree2FDS, getsubtreeBySLT
import re
import eventlet
import time
import os
import re
from neo4jUtil import Neo4j_handle
from collections import Counter
import sys
import pandas as pd
import matplotlib.pyplot as plt
import math
from py2neo import Graph, Node, Relationship, NodeMatcher
from py2neo.ogm import GraphObject, Property, Repository, RelatedFrom
from MathUtil import getSim_HFS, getSim_DHFS, getNDCG, getAP2, getHFSsim, getFormularAttr,getAP
from ast import literal_eval
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils import getFilenameID, reserve_chinese, getContentSimscore, MemTit, MemExprByNum, MemExprByNum1, MemKey, \
    MemRef, \
    MemExprByDHFS, MemERr
import time
import io
from TangentS.math_tan.math_extractor import MathExtractor
from formularRetrieval2 import get_evaluate2

# import logging

# logging.basicConfig(filename='retrieval.log',level=logging.DEBUG,filemode='a')

handler = Neo4j_handle()

repo = Repository("http://localhost:7474")


# graph=Graph("http://localhost:7474")
class Formular(GraphObject):
    __primarykey__ = 'name'
    FormularFDS = Property()
    name = Property()
    formularNum = Property()
    formularSubTreeAttr = Property()
    id = Property()
    title = RelatedFrom("Title", "EXIST")


a = list(repo.match(Formular))


class Title(GraphObject):
    name = Property()
    author = RelatedFrom("Author", "WRITE")
    keyword = RelatedFrom("Keyword", "HAVE")
    Journal = RelatedFrom("Journal", "PUBLISH")
    Reference = RelatedFrom("Reference", "CITE")
    formular = RelatedFrom("Formular", "EXIST")


# sys.exit(0)


def search(filename):
    """
    在数学公式集中查找指定数学公式，获取相似度分数，排名
    :param filename:
    :return: score排名
    """
    print(time.asctime(time.localtime(time.time())) + " HFS搜索数学公式..." + filename)
    # 提取mathml格式格式
    c = open(filename, encoding="utf-8", errors='ignore').read()
    contentForMath = re.sub(r"<mstyle[\s\S]*?>", "", c)
    contentForMath = re.sub(r"</mstyle>", "", contentForMath)
    # 获取公式结构树
    formularTree = MathExtractor.parse_from_xml(contentForMath, 1)
    # 将公式结构树转为FDS结构
    # formularFDS = tree2FDS(formularTree[0])
    subtrees = getFormularAttr(getsubtreeBySLT(formularTree[0]))

    score = {}
    for i in range(len(a)):
        try:
            qq = a[i]
            ww = qq.formularSubTreeAttr
            s = getHFSsim(list(subtrees.values()), literal_eval(a[i].formularSubTreeAttr), i)
            score[a[i].formularNum] = s
        except Exception as e:
            # print(e)
            continue
            # try:
            # print(a[i].FormularFDS)
            # print(time.asctime(time.localtime(time.time()))+" 计算 "+a[i].formularNum+" 相似度出错")
            # print(literal_eval(a[i].FormularFDS))
            # except:
            #     print(time.asctime(time.localtime(time.time()))+" 计算 None 相似度出错")
    # print("HFS Sorted before: " + score)
    # s=sorted(score.items(), key=lambda x: x[1], reverse=True)
    # print("HFS Sorted before: " + s)
    return score


def searchBySubformula(filename):
    print(time.asctime(time.localtime(time.time())) + " DHFS搜索数学公式..." + filename)
    # 提取mathml格式格式
    c = open(filename, encoding="utf-8", errors='ignore').read()
    contentForMath = re.sub(r"<mstyle[\s\S]*?>", "", c)
    contentForMath = re.sub(r"</mstyle>", "", contentForMath)
    # 获取公式结构树
    formularTree = MathExtractor.parse_from_xml(contentForMath, 1)
    formularFDS = tree2FDS(formularTree[0])
    subtrees = getsubtreeBySLT(formularTree)
    score = {}
    for i in range(len(a)):
        try:
            s = getSim_DHFS(formularFDS, literal_eval(a[i].FormularFDS), i)
            s1 = getSim_HFS(formularFDS, literal_eval(a[i].FormularFDS), i)
            score[a[i].formularNum] = s
        except Exception as e:
            continue
            # try:
            # print(a[i].FormularFDS)
            # print(time.asctime(time.localtime(time.time()))+" 计算 "+a[i].formularNum+" 相似度出错")
            # print(literal_eval(a[i].FormularFDS))
            # except:
            #     print(time.asctime(time.localtime(time.time()))+" 计算 None 相似度出错")
    # print("HFS Sorted before: " + score)
    # s=sorted(score.items(), key=lambda x: x[1], reverse=True)
    # print("HFS Sorted before: " + s)
    return score


def semSim(filename):
    print(time.asctime(time.localtime(time.time())) + " 语义相似 搜索数学公式..." + filename)
    queryName = 'query:' + filename.split('/')[1].split('.')[0][5:]
    # 语义嵌入

    # 获取查询的嵌入
    d = os.path.abspath(os.path.dirname(__file__))
    q_address = os.path.join(d + '/../qqqq.npy')
    queryEmbeddings = np.load(q_address, allow_pickle=True).item()

    queryEmbedding = queryEmbeddings[queryName]

    # 数据集
    temp_address = os.path.join(d + '/../eeee.npy')
    data = np.load(temp_address, allow_pickle=True).item()

    score = {}
    for dEmbedding in data.keys():
        q = np.array(queryEmbedding).reshape(1, -1)
        d = np.array(data[dEmbedding]).reshape(1, -1)
        sim = cosine_similarity(q, d)
        a = sim.tolist()[0][0]
        score[dEmbedding] = a

    # s = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return score


def searchFormular(filename):
    s1 = search(filename)
    s2 = semSim(filename)
    a1 = sorted(s1.items(), key=lambda x: x[1], reverse=True)
    a2 = sorted(s2.items(), key=lambda x: x[1], reverse=True)
    # get_ndcg(str(filename)+" HFS:",s1)
    # get_ndcg(str(filename)+" ESIM:", s2)

    s = {}
    print(time.asctime(time.localtime(time.time())) + " DHFS与语义相似 总得分..." + filename)

    for f in s1.keys():
        s[f] = max(s1[f], s2[f])

    # print(a1[:10])
    # print(a2[:10])
    sortedS = sorted(s.items(), key=lambda x: x[1], reverse=True)
    printf(filename,"HFS+Embedding",sortedS[:5])
    return s


def getContentSim(formularNum):
    print(time.asctime(time.localtime(time.time())) + "获取 " + str(formularNum) + " 所在文件内容")
    file = formularNum[:formularNum.index(":")]
    try:
        filename = handler.get_property("Title", file, "fileDir")[0]["n.fileDir"]
    except Exception as e:
        # print(e)
        # print(time.asctime(time.localtime(time.time())) + "获取 " + str(formularNum) + " 所在文件内容 出错")
        return
    name = handler.get_property("Title", file, "name")[0]["n.name"]
    queryConetent = reserve_chinese(
        BeautifulSoup(open(filename, encoding="utf-8", errors='ignore')).find('div', class_='content').text)
    s = {}
    dirDist = handler.get_all_Title()
    i = 0
    for dir in dirDist:
        i += 1
        if MemTit(filename, dir["file_dir"]) == 0 or MemTit(name, dir["name"]) == 0: continue
        conetent = reserve_chinese(
            BeautifulSoup(open(dir["file_dir"], encoding="utf-8", errors='ignore')).find('div', class_='content').text)
        contentSim = getContentSimscore(queryConetent, conetent)
        # print(time.asctime(time.localtime(time.time())) +" "+str(i)+" : "+str(contentSim))
        f = getFilenameID(dir["file_dir"])
        s[f] = contentSim
    # score = sorted(s.items(), key=lambda x: x[1])
    # print(score)
    # return score
    return s


def getSubGraph(formularNum):
    """
    通过指定的数学公式 在 集 中的编号，查找对应的子图
    :param formularNum:
    :return:
    """
    # eventlet.monkey_patch()
    tag = 0
    # with eventlet.Timeout(60 * 5, False):
    tag = 1
    print(time.asctime(time.localtime(time.time())) + " 获取 " + str(formularNum) + " 所在子图")
    querySubGraph = handler.get_Titles_By_Formular(formularNum)
    # 分析重复标题，只存一次
    titleList = set()
    for tempNode in querySubGraph:
        tempa = dict(tempNode['m'])
        titleList.add(tempa['name'])
    # 找到公式所在子图的标题（会有很多个）
    print("111")
    # 找到标题所在的子图

    # 查询子图匹配
    # 先将原始查询划分为多个子查询
    queryTitle = ""
    queryReference = []
    queryKeyword = []
    queryAuthor = []
    queryFormular = []
    queryJournal = None
    for queryNode in querySubGraph:
        d = queryNode['node']
        s = str(d.labels)[1:]
        if s == "Formular":
            try:
                queryFormular.append(dict(d)['FormularFDS'])
            except:
                # print("")
                continue
        elif s == "Title":
            queryTitle = dict(d)['name']
        elif s == "Author":
            queryAuthor.append(dict(d)['name'])
        elif s == "Journal":
            queryJournal = dict(d)['name']
        elif s == "Keyword":
            queryKeyword.append(dict(d)['name'])
        elif s == "Reference":
            queryReference.append(dict(d)['name'])
    tag = 0
    if tag == 1: return None
    sco1 = {}
    sco2 = {}
    sco = {}
    # 计算隶属度

    titles = handler.get_entity("Title")
    p = 0
    for t in titles:
        p += 1
        # print(time.asctime(time.localtime(time.time())) + " 计算 " + str(p) +" : "+str(dict(t['n'])['name']) + " 与给定子图相似度")
        tit = dict(t['n'])['name']
        subGraphCandi = handler.get_subgraphNodes_By_Title(tit)
        ref = []
        key = []
        aut = []
        j = None
        expr = []
        cnt = 0
        for n in subGraphCandi:
            d = n['node']
            s = str(d.labels)[1:]
            if s == "Author":
                aut.append(dict(d)['name'])
            elif s == "Keyword":
                key.append(dict(d)['name'])
            elif s == "Journal":
                j = dict(d)['name']
            elif s == "Formular":
                try:
                    expr.append(dict(d)['FormularFDS'])
                except:
                    # print(time.asctime(time.localtime(time.time())) + "当前数学公式没有FDS属性")
                    continue
            elif s == "Reference":
                ref.append(dict(d)['name'])
        # 过滤

        tD = MemTit(queryTitle, tit)
        tA = MemKey(queryAuthor, aut)
        tK = MemKey(queryKeyword, key)
        tJ = MemTit(queryJournal, j)
        if [tD, tA, tK, tJ].count(0) > 2: continue
        tF = MemExprByNum(queryFormular, expr)
        # if tF==0:continue
        MemExprByNum1(queryFormular, expr)
        tR = MemRef(queryReference, ref)
        Ua = (tD, tA, tK, tJ, tF, tR)

        # Us=(tD,tA,tK,tJ,tR)
        sim1 = (tD + tA + tK + tJ + tF + tR) / 6
        print("title:" + tit + " : " + str(Ua) + " -> " + str((tD + tA + tK + tJ) / 4))
        # 边松弛比

        bA = MemERr(queryAuthor, aut)
        bK = MemERr(queryKeyword, key)
        bR = MemERr(queryReference, ref)
        bE = MemERr(queryFormular, expr)
        Ub = (1, bA, bK, 1, bE, bR)
        sim2 = (1 + bA + bK + 1 + bE + bR) / 6
        Na = np.array(Ua)
        Nb = np.array(Ub)

        # todo:计算权重
        # 计算最优最差方案
        # Umax = np.maximum(Na, Nb)
        # Umin = np.minimum(Na, Nb)
        # 计算内容与结构方案与最优最差方案的距离（欧氏距离
        # SmaxA = np.sqrt(np.sum(np.square(Na - Umax)))
        # SmaxB = np.sqrt(np.sum(np.square(Nb - Umax)))
        # SminA = np.sqrt(np.sum(np.square(Na - Umin)))
        # SminB = np.sqrt(np.sum(np.square(Nb - Umin)))
        # 计算接近度C
        # C_A = SminA / (SmaxA + SminA)
        # C_B = SminB / (SminB + SmaxB)
        # 对C加权求和得到结果
        # sim = C_A * 0.8 + 0.5 * C_B

        sco1[tit] = sim1
        sco2[tit] = sim2
        sco[tit] = sim1 * 0.8 + sim2 * 0.5

    score1 = sorted(sco1.items(), key=lambda x: x[1], reverse=True)

    score2 = sorted(sco2.items(), key=lambda x: x[1], reverse=True)
    score = sorted(sco.items(), key=lambda x: x[1], reverse=True)
    printf(formularNum, "MAXSIM", score[0][1])
    return sco


def getSubGraph2(formularNum):
    """
    通过指定的数学公式 在 集 中的编号，查找对应的子图
    :param formularNum:
    :return:
    """
    # sco1 = {}
    # sco2 = {}
    sco = {}
    eventlet.monkey_patch()
    tag = 0
    with eventlet.Timeout(60 * 5, False):
        tag = 1
        print(time.asctime(time.localtime(time.time())) + " 获取 " + str(formularNum) + " 所在子图")
        querySubGraph = handler.get_Titles_By_Formular(formularNum)
        # 分析重复标题，只存一次
        # 找到公式所在子图的标题（会有很多个）
        titleList = set()
        for tempNode in querySubGraph:
            tempa = dict(tempNode['m'])
            titleList.add(tempa['name'])
        tag = 0
    if tag == 1: return None

    # 找到标题所在的子图
    for queryTitle in titleList:
        querySubGraph = handler.get_subgraphNodes_By_Title(queryTitle)
        # 查询子图匹配
        # 先将原始查询划分为多个子查询
        queryTitle = ""
        queryReference = []
        queryKeyword = []
        queryAuthor = []
        queryFormular = []
        queryJournal = None
        for queryNode in querySubGraph:
            d = queryNode['node']
            s = str(d.labels)[1:]
            if s == "Formular":
                try:
                    queryFormular.append(dict(d)['FormularFDS'])
                except:
                    # print("")
                    continue
            elif s == "Title":
                queryTitle = dict(d)['name']
            elif s == "Author":
                queryAuthor.append(dict(d)['name'])
            elif s == "Journal":
                queryJournal = dict(d)['name']
            elif s == "Keyword":
                queryKeyword.append(dict(d)['name'])
            elif s == "Reference":
                queryReference.append(dict(d)['name'])

        # 计算隶属度

        titles = handler.get_entity("Title")
        p = 0
        for t in titles:
            p += 1
            # print(time.asctime(time.localtime(time.time())) + " 计算 " + str(p) +" : "+str(dict(t['n'])['name']) + " 与给定子图相似度")
            tit = dict(t['n'])['name']
            subGraphCandi = handler.get_subgraphNodes_By_Title(tit)
            ref = []
            key = []
            aut = []
            j = None
            expr = []
            cnt = 0
            for n in subGraphCandi:
                d = n['node']
                s = str(d.labels)[1:]
                if s == "Author":
                    aut.append(dict(d)['name'])
                elif s == "Keyword":
                    key.append(dict(d)['name'])
                elif s == "Journal":
                    j = dict(d)['name']
                elif s == "Formular":
                    try:
                        expr.append(dict(d)['FormularFDS'])
                    except:
                        # print(time.asctime(time.localtime(time.time())) + "当前数学公式没有FDS属性")
                        continue
                elif s == "Reference":
                    ref.append(dict(d)['name'])
            # 过滤

            tD = MemTit(queryTitle, tit)
            tA = MemKey(queryAuthor, aut)
            tK = MemKey(queryKeyword, key)
            tJ = MemTit(queryJournal, j)
            if [tD, tA, tK, tJ].count(0) > 2: continue
            tF = MemExprByNum(queryFormular, expr)
            # if tF==0:continue
            MemExprByNum1(queryFormular, expr)
            tR = MemRef(queryReference, ref)
            Ua = (tD, tA, tK, tJ, tF, tR)
            sim1 = (tD + tA + tK + tJ + tF + tR) / 6
            print("title:" + tit + " : " + str(Ua) + " -> " + str((tD + tA + tK + tJ) / 4))
            # 边松弛比

            bA = MemERr(queryAuthor, aut)
            bK = MemERr(queryKeyword, key)
            bR = MemERr(queryReference, ref)
            bE = MemERr(queryFormular, expr)
            sim2 = (1 + bA + bK + 1 + bE + bR) / 6
            # sco1[tit] = sim1
            # sco2[tit] = sim2
            if tit in sco:
                sco[tit] = max(sco[tit], sim1 * 0.8 + sim2 * 0.5)
            else:
                sco[tit] = sim1 * 0.8 + sim2 * 0.5

    # score1 = sorted(sco1.items(), key=lambda x: x[1], reverse=True)

    # score2 = sorted(sco2.items(), key=lambda x: x[1], reverse=True)
    score = sorted(sco.items(), key=lambda x: x[1], reverse=True)
    printf(formularNum, "MAXSIM", score[0][1])
    return sco


def getSubGraph3(queryTitles):
    """
    通过指定的标题，查找对应的子图
    :param formularNum:
    :return:
    """
    sco = {}
    # 找到标题所在的子图
    for queryTitle in queryTitles:
        print(time.asctime(time.localtime(time.time())) + " 获取 " + str(queryTitle) + " 所在子图")
        querySubGraph = handler.get_subgraphNodes_By_Filename(queryTitle)
        # 查询子图匹配
        # 先将原始查询划分为多个子查询
        queryTitle = ""
        queryReference = []
        queryKeyword = []
        queryAuthor = []
        queryFormular = []
        queryJournal = None
        for queryNode in querySubGraph:
            d = queryNode['node']
            s = str(d.labels)[1:]
            if s == "Formular":
                try:
                    queryFormular.append(dict(d)['FormularFDS'])
                except:
                    # print("")
                    continue
            elif s == "Title":
                queryTitle = dict(d)['name']
            elif s == "Author":
                queryAuthor.append(dict(d)['name'])
            elif s == "Journal":
                queryJournal = dict(d)['name']
            elif s == "Keyword":
                queryKeyword.append(dict(d)['name'])
            elif s == "Reference":
                queryReference.append(dict(d)['name'])

        # 计算隶属度

        titles = handler.get_entity("Title")
        p = 0
        for t in titles:
            p += 1
            # print(time.asctime(time.localtime(time.time())) + " 计算 " + str(p) +" : "+str(dict(t['n'])['name']) + " 与给定子图相似度")
            tit = dict(t['n'])['name']
            subGraphCandi = handler.get_subgraphNodes_By_Title(tit)
            ref = []
            key = []
            aut = []
            j = None
            expr = []
            cnt = 0
            for n in subGraphCandi:
                d = n['node']
                s = str(d.labels)[1:]
                if s == "Author":
                    aut.append(dict(d)['name'])
                elif s == "Keyword":
                    key.append(dict(d)['name'])
                elif s == "Journal":
                    j = dict(d)['name']
                elif s == "Formular":
                    try:
                        expr.append(dict(d)['FormularFDS'])
                    except:
                        # print(time.asctime(time.localtime(time.time())) + "当前数学公式没有FDS属性")
                        continue
                elif s == "Reference":
                    ref.append(dict(d)['name'])
            # 过滤

            tD = MemTit(queryTitle, tit)
            tA = MemKey(queryAuthor, aut)
            tK = MemKey(queryKeyword, key)
            tJ = MemTit(queryJournal, j)
            if [tD, tA, tK, tJ].count(0) > 2: continue
            tF = MemExprByNum(queryFormular, expr)
            # if tF==0:continue
            MemExprByNum1(queryFormular, expr)
            tR = MemRef(queryReference, ref)
            Ua = (tD, tA, tK, tJ, tF, tR)
            sim1 = (tD + tA + tK + tJ + tF + tR) / 6
            print("title:" + tit + " : " + str(Ua) + " -> " + str((tD + tA + tK + tJ) / 4))
            # 边松弛比

            bA = MemERr(queryAuthor, aut)
            bK = MemERr(queryKeyword, key)
            bR = MemERr(queryReference, ref)
            bE = MemERr(queryFormular, expr)
            sim2 = (1 + bA + bK + 1 + bE + bR) / 6
            # sco1[tit] = sim1
            # sco2[tit] = sim2
            if tit in sco:
                sco[tit] = max(sco[tit], sim1 * 0.8 + sim2 * 0.5)
            else:
                sco[tit] = sim1 * 0.8 + sim2 * 0.5

    # score1 = sorted(sco1.items(), key=lambda x: x[1], reverse=True)

    # score2 = sorted(sco2.items(), key=lambda x: x[1], reverse=True)
    score = sorted(sco.items(), key=lambda x: x[1], reverse=True)
    return sco


def printf(name, method, data):
    res = open('result.txt', 'a')
    print(time.asctime(time.localtime(time.time())) + " " + str(name) + "【" + method + "】" + str(data))
    res.write(time.asctime(time.localtime(time.time())) + " " + str(name) + "【" + method + "】" + str(data) + "\n")
    res.close()


def get_evaluate(name, r):
    get_NDCG(name, r)
    get_AP(name, r)



def get_AP(name, r):
    a5 = getAP(r, 5)
    printf(name, "AP5", a5)
    a10 = getAP(r, 10)
    printf(name, "AP10", a10)
    a15 = getAP(r, 15)
    printf(name, "AP15", a15)
    a20 = getAP(r, 20)
    printf(name, "AP20", a20)

def get_AP2(name, r):
    a5 = getAP2(r, 5)
    printf(name, "AP5", a5)
    a10 = getAP2(r, 10)
    printf(name, "AP10", a10)
    a15 = getAP2(r, 15)
    printf(name, "AP15", a15)
    a20 = getAP2(r, 20)
    printf(name, "AP20", a20)


def get_NDCG(name, r):
    n5 = getNDCG(r, 5)
    printf(name, "NDCG5", n5)
    n10 = getNDCG(r, 10)
    printf(name, "NDCG10", n10)
    n15 = getNDCG(r, 15)
    printf(name, "NDCG15", n15)
    n20 = getNDCG(r, 20)
    printf(name, "NDCG20", n20)


def formularTest():
    # r = searchFormular("query/query10.html")
    # print(r)
    # s=sorted(r.items(), key=lambda x: x[1], reverse=True)
    # get_ndcg(str("query/query10.html")+" HFS:",r)
    for file in os.listdir("query"):
        # if file != "query17.html": continue
        if file not in {
                "query11.html",
                "query12.html",
                "query13.html",
                "query14.html",
                "query15.html",
                "query16.html",
                "query17.html",
                "query18.html",
                "query19.html",
                "query20.html",
                        }: continue
        filename = "query" + "/" + file
        r = searchFormular(filename)
        get_evaluate2("【" + str(filename) + "】 HFS+ESIM:", r)
        # sr = sorted(r.items(), key=lambda x: x[1], reverse=True)
        # subGra=getSubGraph(sr[0][0])
        # get_evaluate(sr[0][1],subGra)


def subGraphTest():
    querySubGraph = {
        "J201936-07-169-176:78",  # 公式多次出现
        'J201936-07-300-306:63',
        'J201936-03-89-95+103:17',
        'J201936-08-247-252:0',
        'J201936-06-277-281+316:3',
        'J201939-05-1357-1363:10',
        'J2019-09-30-30+32:6',
        'J201936-05-298-303+333:0',
        'J201936-11-280-285:16',
        'J201939-07-338-344:7',
        "J201936-03-89-95+103:0",
        "J201936-06-277-281+316:0",
        "J201936-07-110-116:2",
        "J201936-11-121-126+209:0",
        "J201936-10-216-221:0",
        "J201936-07-307-310+316:15",
        "J201936-11-70-77:19",
        "J201936-08-320-324+333:41",
        "J201936-07-300-306:76",
        "J201936-08-172-176+261:0",
        "J201936-08-120-124:0"

    }

    for query in querySubGraph:
        res = getSubGraph2(query)
        if res is None: continue
        get_evaluate(query, res)


def formularTest2():
    for file in os.listdir("query"):
        if file != "query11.html": continue
        # if file not in {"query11.html", "query12.html", "query13.html","query14.html", "query15.html", "query16.html", "query17.html",
        #                 "query18.html", "query19.html", "query20.html", }: continue
        filename = "query" + "/" + file
        r = searchFormular(filename)
        get_evaluate("【" + str(filename) + "】 HFS+ESIM:", r)
        sr = sorted(r.items(), key=lambda x: x[1], reverse=True)
        printf(filename, "公式检索结果", sr[:50])
        max = sr[0][1]
        i = 0
        queryTitles = set()
        while (sr[i][1] == max):
            queryTitles.add(sr[i][0][:sr[i][0].find(":")])
            i += 1
        res = getSubGraph3(queryTitles)
        if res is None: continue
        get_evaluate(filename, res)


def MemExpr(queryFormularAttr, formularsSubTreeAttrs):
    max = 0
    for formular in formularsSubTreeAttrs:
        sim = getHFSsim(queryFormularAttr, formular)
        if sim > max: max = sim
    return max


def formularAndSubGraph(file, title):
    """
    :param file: 格式query11.html
    :return:sco
    """
    sco1 = {}
    sco2 = {}
    sco = {}
    filename = "query" + "/" + file
    # 提取查询表达式
    c = open(filename, encoding="utf-8", errors='ignore').read()
    contentForMath = re.sub(r"<mstyle[\s\S]*?>", "", c)
    contentForMath = re.sub(r"</mstyle>", "", contentForMath)
    # 获取公式结构树
    formularTree = MathExtractor.parse_from_xml(contentForMath, 1)
    subtrees = getFormularAttr(getsubtreeBySLT(formularTree[0]))
    queryFormularAttr = list(subtrees.values())

    # 查询子图解析
    querySubGraph = handler.get_subgraphNodes_By_Title(title)
    queryTitle = ""
    queryReference = []
    queryKeyword = []
    queryAuthor = []
    queryJournal = None
    for queryNode in querySubGraph:
        d = queryNode['node']
        s = str(d.labels)[1:]
        if s == "Title":
            queryTitle = dict(d)['name']
        elif s == "Author":
            queryAuthor.append(dict(d)['name'])
        elif s == "Journal":
            queryJournal = dict(d)['name']
        elif s == "Keyword":
            queryKeyword.append(dict(d)['name'])
        elif s == "Reference":
            queryReference.append(dict(d)['name'])

    # 遍历所有子图，只是公式相似度计算采用的是与查询公式的HFS值
    titles = handler.get_entity("Title")
    p = 0
    for t in titles:
        p += 1
        # print(time.asctime(time.localtime(time.time())) + " 计算 " + str(p) +" : "+str(dict(t['n'])['name']) + " 与给定子图相似度")
        tit = dict(t['n'])['name']
        subGraphCandi = handler.get_subgraphNodes_By_Title(tit)
        ref = []
        key = []
        aut = []
        j = None
        expr = []
        cnt = 0
        for n in subGraphCandi:
            d = n['node']
            s = str(d.labels)[1:]
            if s == "Author":
                aut.append(dict(d)['name'])
            elif s == "Keyword":
                key.append(dict(d)['name'])
            elif s == "Journal":
                j = dict(d)['name']
            elif s == "Formular":
                try:
                    expr.append(literal_eval(dict(d)['formularSubTreeAttr']))
                except:
                    # print(time.asctime(time.localtime(time.time())) + "当前数学公式没有FDS属性")
                    continue
            elif s == "Reference":
                ref.append(dict(d)['name'])
        # 过滤

        tD = MemTit(queryTitle, tit)
        tA = MemKey(queryAuthor, aut)
        tK = MemKey(queryKeyword, key)
        tJ = MemTit(queryJournal, j)
        if [tD, tA, tK, tJ].count(0) > 2: continue
        tF = MemExpr(queryFormularAttr, expr)
        tR = MemRef(queryReference, ref)
        Ua = (tD, tA, tK, tJ, tF, tR)
        sim1 = (tD + tA + tK + tJ + tF + tR) / 6
        print("title:" + tit + " : " + str(Ua) + " -> " + str((tD + tA + tK + tJ) / 4))
        # 边松弛比

        bA = MemERr(queryAuthor, aut)
        bK = MemERr(queryKeyword, key)
        bR = MemERr(queryReference, ref)
        bE = 1
        sim2 = (1 + bA + bK + 1 + bE + bR) / 6
        sco1[tit] = sim1
        sco2[tit] = sim2
        sco[tit] = sim1 * 0.8 + sim2 * 0.5

    score1 = sorted(sco1.items(), key=lambda x: x[1], reverse=True)
    score2 = sorted(sco2.items(), key=lambda x: x[1], reverse=True)
    score = sorted(sco.items(), key=lambda x: x[1], reverse=True)
    # printf("formularAndSubGraph","内容隶属度排序结果",score1[:50])
    # printf("formularAndSubGraph", "边比隶属度排序结果", score2[:50])
    # printf("formularAndSubGraph", "隶属度排序结果", score[:50])
    return sco


if __name__ == '__main__':
    # subGraphTest()
    formularTest()
    # formularAndSubGraph("query11.html","考虑擦除编码可靠视频流三步式内凸逼近优化")
