#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/14 15:50
# @Author  : LiXin
# @File    : formularRetrieval.py
# @Describe:检索
from bs4 import BeautifulSoup
from TangentS.math_tan.math_extractor import MathExtractor
from MathUtil import tree2FDS, getsubtreeBySLT
import os
import re
from neo4jUtil import Neo4j_handle
from MathUtil import getNDCG, getAP2, getHFSsim, getFormularAttr, getNDCG2
from ast import literal_eval
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import MemTit, MemKey, MemRef, MemERr
import time
from TangentS.math_tan.math_extractor import MathExtractor

handler = Neo4j_handle()


def printf(name, method, data):
    res = open('result.txt', 'a')
    print(time.asctime(time.localtime(time.time())) + " " + str(name) + "【" + method + "】" + str(data))
    res.write(time.asctime(time.localtime(time.time())) + " " + str(name) + "【" + method + "】" + str(data) + "\n")
    res.close()


def get_evaluate2(name, r):
    get_NDCG2(name, r)
    get_AP2(name, r)


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


def get_NDCG2(name, r):
    n5 = getNDCG2(r, 5)
    printf(name, "NDCG5", n5)
    n10 = getNDCG2(r, 10)
    printf(name, "NDCG10", n10)
    n15 = getNDCG2(r, 15)
    printf(name, "NDCG15", n15)
    n20 = getNDCG2(r, 20)
    printf(name, "NDCG20", n20)


def MemExpr(formularNum, queryFormularAttr, formulars):
    maxSIM = 0

    d = os.path.abspath(os.path.dirname(__file__))
    # 数据集
    temp_address = os.path.join(d + '/../eeee.npy')
    data = np.load(temp_address, allow_pickle=True).item()
    q = np.array(data[formularNum]).reshape(1, -1)
    for formular in formulars:
        # HFSSIM
        sim1 = getHFSsim(queryFormularAttr, formulars[formular])
        # EmbeddingSIM
        d = np.array(data[formular]).reshape(1, -1)
        sim2 = cosine_similarity(q, d)
        sim2 = sim2.tolist()[0][0]
        maxSIM = max(sim1, sim2, maxSIM)

    return maxSIM


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
        if tit == queryTitle: continue
        subGraphCandi = handler.get_subgraphNodes_By_Title(tit)
        ref = []
        key = []
        aut = []
        j = None
        expr = {}
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
                    if dict(d)['formularNum'] not in expr.keys():
                        expr[dict(d)['formularNum']] = literal_eval(dict(d)['formularSubTreeAttr'])
                except:
                    try:
                        if dict(d)['formularNum'] in expr.keys():
                            expr.pop(dict(d)['formularNum'])
                    # print(time.asctime(time.localtime(time.time())) + "当前数学公式没有FDS属性")
                    except:
                        continue
                    continue
            elif s == "Reference":
                ref.append(dict(d)['name'])
        # 过滤

        tD = MemTit(queryTitle, tit)
        tA = MemKey(queryAuthor, aut)
        tK = MemKey(queryKeyword, key)
        tJ = MemTit(queryJournal, j)
        if [tD, tA, tK, tJ].count(0) > 2: continue
        tF = MemExpr(filename, queryFormularAttr, expr)
        tR = MemRef(queryReference, ref)
        Ua = (tD, tA, tK, tJ, tF, tR)
        sim1 = (tD + tA + tK + tJ + tF + tR) / 6

        # 边松弛比

        bA = MemERr(queryAuthor, aut)
        bK = MemERr(queryKeyword, key)
        bR = MemERr(queryReference, ref)
        bE = 1
        sim2 = (1 + bA + bK + 1 + bE + bR) / 6

        sco1[tit] = sim1
        sco2[tit] = sim2
        sco[tit] = sim1 * 0.8 + sim2 * 0.5
        print(time.asctime(time.localtime(time.time())) + "SubGraph " + " SIM " + " title: " + tit + " : " + str(
            sim1) + " -- " + str(sim2) + " -- " + str(
            sim1 * 0.8 + sim2 * 0.5))

    score1 = sorted(sco1.items(), key=lambda x: x[1], reverse=True)
    score2 = sorted(sco2.items(), key=lambda x: x[1], reverse=True)
    score = sorted(sco.items(), key=lambda x: x[1], reverse=True)
    printf("formularAndSubGraph", "内容隶属度排序结果", score1[:50])
    printf("formularAndSubGraph", "边比隶属度排序结果", score2[:50])
    printf("formularAndSubGraph", "隶属度排序结果", score[:50])
    return sco


def formularAndSubGraph2(formularNum, fileName):
    sco1 = {}
    sco2 = {}
    sco = {}
    queryFormularAttr = handler.get_formularSubTreeAttr_By_FormularNum(formularNum)
    queryFormularAttr = literal_eval(queryFormularAttr[0]['n.formularSubTreeAttr'])
    # 查询子图解析
    querySubGraph = handler.get_subgraphNodes_By_Title(fileName)
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
        if tit == queryTitle: continue
        subGraphCandi = handler.get_subgraphNodes_By_Title(tit)
        ref = []
        key = []
        aut = []
        j = None
        expr = {}
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
                    if dict(d)['formularNum'] not in expr.keys():
                        expr[dict(d)['formularNum']] = literal_eval(dict(d)['formularSubTreeAttr'])
                except:
                    try:
                        if dict(d)['formularNum'] in expr.keys():
                            expr.pop(dict(d)['formularNum'])
                    # print(time.asctime(time.localtime(time.time())) + "当前数学公式没有FDS属性")
                    except:
                        continue
                    continue
            elif s == "Reference":
                ref.append(dict(d)['name'])
        # 过滤

        tD = MemTit(queryTitle, tit)
        tA = MemKey(queryAuthor, aut)
        tK = MemKey(queryKeyword, key)
        tJ = MemTit(queryJournal, j)
        if [tD, tA, tK, tJ].count(0) > 2: continue
        tF = MemExpr(formularNum, queryFormularAttr, expr)
        tR = MemRef(queryReference, ref)
        Ua = (tD, tA, tK, tJ, tF, tR)
        sim1 = (tD + tA + tK + tJ + tF + tR) / 6

        # 边松弛比

        bA = MemERr(queryAuthor, aut)
        bK = MemERr(queryKeyword, key)
        bR = MemERr(queryReference, ref)
        bE = 1
        sim2 = (1 + bA + bK + 1 + bE + bR) / 6

        sco1[tit] = sim1
        sco2[tit] = sim2
        sco[tit] = sim1 * 0.8 + sim2 * 0.5
        print(time.asctime(time.localtime(time.time())) +
              "SubGraph " + " SIM " + " title: " + tit + str(Ua) + " : " + str(sim1) + " -- " + str(
            sim2) + " -- " + str(
            sim1 * 0.8 + sim2 * 0.5))

    score1 = sorted(sco1.items(), key=lambda x: x[1], reverse=True)
    score2 = sorted(sco2.items(), key=lambda x: x[1], reverse=True)
    score = sorted(sco.items(), key=lambda x: x[1], reverse=True)
    printf("formularAndSubGraph", "内容隶属度排序结果", score1[:10])
    # printf("formularAndSubGraph", "边比隶属度排序结果", score2[:50])
    printf("formularAndSubGraph", "隶属度排序结果", score[:10])
    return sco


if __name__ == '__main__':
    # get_evaluate2("query11+考虑擦除编码可靠视频流三步式内凸逼近优化",formularAndSubGraph("query11.html","考虑擦除编码可靠视频流三步式内凸逼近优化"))
    # get_evaluate2("J2019-21-141-141+143:0浅谈泰勒公式在高考数学压轴题中的应用",formularAndSubGraph2("J2019-21-141-141+143:0", "浅谈泰勒公式在高考数学压轴题中的应用"))
    # get_evaluate2("J201936-10-216-221:2基于高斯拉普拉斯算子的多聚焦图像融合",formularAndSubGraph2("J201936-10-216-221:2", "基于高斯拉普拉斯算子的多聚焦图像融合"))#
    # get_evaluate2("J2019-20-11-14:0对数学运算的认识与理解",formularAndSubGraph2("J2019-20-11-14:0", "对数学运算的认识与理解"))
    # get_evaluate2("J201938-18-270-276:1冲击应力对电连接器性能影响的仿真研究", formularAndSubGraph2("J201938-18-270-276:1", "冲击应力对电连接器性能影响的仿真研究"))
    # get_evaluate2("J201958-07-46-49:84直观助思考思辨破难题——2017年新课标Ⅰ导数压轴题剖析及启示",formularAndSubGraph2("J201958-07-46-49:84","直观助思考  思辨破难题——2017年新课标Ⅰ导数压轴题剖析及启示"));
    # get_evaluate2("J201942-15-12-17:11基于BPNN的机动识别方法",formularAndSubGraph2("J201942-15-12-17:11", "基于BPNN的机动识别方法"))
    # get_evaluate2("J201950-07-153-159:2风积沙改性土热物理性质的测试与分析", formularAndSubGraph2("J201950-07-153-159:2", "风积沙改性土热物理性质的测试与分析 "))
    # get_evaluate2("J201936-09-126-130:0基于动态神经网络的电极式锅炉数学建模研究",formularAndSubGraph2("J201936-09-126-130:0", "基于动态神经网络的电极式锅炉数学建模研究"))
    # get_evaluate2("J201936-09-81-84+180:12基于人工蜂群算法的分布式入侵攻击检测系统",formularAndSubGraph2("J201936-09-81-84+180:12", "基于人工蜂群算法的分布式入侵攻击检测系统"))
    get_evaluate2("J201936-06-243-247:8双重不确定分数阶混沌系统的鲁棒自适应同步控制算法研究",formularAndSubGraph2("J201936-06-243-247:8", "双重不确定分数阶混沌系统的鲁棒自适应同步控制算法研究"))
    get_evaluate2("J2019-03-60-62:1精致概念  刨根问底——以“一道单元测试题”为例",formularAndSubGraph2("J2019-03-60-62:1", "精致概念  刨根问底——以“一道单元测试题”为例"))
