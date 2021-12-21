#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/15 20:29
# @Author  : LiXin
# @File    : dataExtr.py
# @Describe:

from bs4 import BeautifulSoup
from neo4jUtil import Neo4j_handle
from py2neo import Node, Relationship
from TangentS.math_tan.math_extractor import MathExtractor
from MathUtil import tree2FDS,getHFS,getNDCG
import pickle
import os
import re
import unicodedata
import numpy as np
from utils import reserve_chinese,getContentSimscore
f_id = -1


def extr():
    dir = "中文数据集"
    for fileDir in os.listdir(dir):

        dirChild = dir + "/" + fileDir
        for file in os.listdir(dirChild):
            filename = dirChild + "/" + file
            print("正在创建..." + filename)
            extrSingleFile(filename)


def extrPackage():
    dir = "中文数据集"
    fileDir = "J中国卫生统计"
    dirChild = dir + "/" + fileDir
    cnt = 0
    allScores={}
    #formular:[id,[]] {id,[]}
    for file in os.listdir(dirChild):
        filename = dirChild + "/" + file
        print("正在创建..." + filename)
        scores=extrSingleFile(filename)
        if len(scores) == 0:continue
        allScores.update(scores)
    print(allScores)
    print(sorted(allScores.items(), key=lambda x: x[1], reverse=False))

    # print(getNDCG(allScores,10))
    # print(f_id)
    # print(len(allScores))


queryArray=[]
queryConetent=reserve_chinese(BeautifulSoup(open("中文数据集/J中国卫生统计/J中国卫生统计201936-02-291-294.html", encoding="utf-8", errors='ignore')).find('div',class_='content').text)
def extrSingleFile(filename):
    #filename
    partFilename = filename.split('/')[-1]
    partFilename = os.path.splitext(partFilename)[0]
    temp = str(unicodedata.normalize('NFKD', partFilename).encode('ascii', 'ignore'))
    temp = temp[2:]
    file_name = temp[:-1]

    #数据处理
    soup = BeautifulSoup(open(filename, encoding="utf-8", errors='ignore'))
    # formulars = soup.find_all('math')
    content=reserve_chinese(soup.find('div',class_='content').text)
    contentSim=getContentSimscore(queryConetent,content)
    return {file_name:contentSim}
    contentForMath = re.sub(r"<mstyle[\s\S]*?>", "", open(filename, encoding="utf-8", errors='ignore').read())
    contentForMath = re.sub(r"</mstyle>", "", contentForMath)
    formularTrees = MathExtractor.parse_from_xml(contentForMath, 1)
    title = soup.find('h1').text.replace("\n", "")
    authors = soup.find('h2').text
    authorsLists = authors[1:len(authors) - 1].split('\n')
    journal = soup.find('div', class_='tips').text.replace("\n", "")
    journal = journal[0:journal.index(' ')]
    keyWord = []
    keyWordLists = []
    try:
        keyWord = soup.find('div', attrs={"id": "a_keywords"}).find('p').text
        keyWordLists = keyWord[1:len(keyWord) - 1].split('\n')
    except Exception as e:
        print("[ERROR]创建" + filename + "出错.. 缺少关键字")

    #参考文献
    references=soup.find_all('div',class_='reference anchor-tag')[0].find_all('a')
    referencesList=[]
    #作者，名称，期刊
    for reference in references:
        referenceList = []
        tags = reference.text.replace("\n", "").split('.')
        referenceList.append(tags[0][tags[0].index(' ')+1:].split(','))
        referenceList.append(tags[1])
        referenceList.append(tags[2][:tags[2].index(',')])
        referencesList.append(referenceList)
    print(references)


    # handle=Neo4j_handle()
    #
    #     handle.add_relationship("Title",title,"EXIST","Formular",f,pickle.dump(f))
    # for a in authorsLists:
    #     handle.add_relationship("Title",title,"WRITE","Author",a)
    # handle.add_relationship("Title",title,"PUBLISH","Journal",journal)
    # for k in keyWordLists:
    #     handle.add_relationship("Title",title,"HAVE","Keyword",k)

    # 数学公式MathML --> 结构树 --> FDS解析
    # file=open(filename, encoding="utf-8", errors='ignore')
    # contentForMath=file.read()
    # contentForMath = re.sub(r"<mstyle[\s\S]*?>", "", contentForMath)
    # contentForMath = re.sub(r"</mstyle>", "", contentForMath)
    # formularTrees = MathExtractor.parse_from_xml(contentForMath, 1)
    # if len(formulars) == 0 and len(formularTrees) == 0:
    #     print("没有数学公式")
    #     return
    # if len(formulars) != len(formularTrees):
    #     print("SOS 数目不一样")

    # 数目一样，解析正确
    # 结构树 --> FDS解析

    scores={}
    global queryArray
    for formularTree in formularTrees.values():
        FDSFormularSymbols = tree2FDS(formularTree)
        global f_id
        f_id+=1
        if f_id==0:
            queryArray=FDSFormularSymbols
        #计算犹豫模糊值h(Ul,Un,Ulevel)
        # HFSArray=getHFS(FDSFormularSymbols)
        # 计算犹豫模糊值h(五个隶属度)
        score=getSim(queryArray,FDSFormularSymbols,f_id)
        scores[file_name + ":" +str(f_id)]=score
    return scores

if __name__ == '__main__':
    # extrSingleFile("中文数据集/J中学数学教学/J中学数学教学201901-9-12.html")
    # extr()
    extrPackage()
