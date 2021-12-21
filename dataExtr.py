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
from MathUtil import tree2FDS,getHFS,getNDCG,getsubtreeBySLT,getFormularAttr
import pickle
import os
import re
import unicodedata
import numpy as np
from utils import reserve_chinese,remove_digits
import time
f_id = -1


def extr():
    dir = "中文数据集"
    handle = Neo4j_handle()
    for fileDir in os.listdir(dir):
        dirChild = dir + "/" + fileDir
        for file in os.listdir(dirChild):
            filename = dirChild + "/" + file
            print(time.asctime(time.localtime(time.time()))+" 正在创建..." + filename)
            # extrSingleFile(filename,handle)
            addFormularSubTreeAttr(filename,handle)


def extrPackage():
    dir = "中文数据集"
    fileDir = "J中国卫生统计"
    dirChild = dir + "/" + fileDir
    cnt = 0
    handle = Neo4j_handle()
    #formular:[id,[]] {id,[]}
    for file in os.listdir(dirChild):
        filename = dirChild + "/" + file
        print(time.asctime(time.localtime(time.time()))+" 正在创建..." + filename)
        extrSingleFile(filename,handle)


def extrSingleFile(filename,handle):
    #filename
    partFilename = filename.split('/')[-1]
    partFilename = os.path.splitext(partFilename)[0]
    temp = str(unicodedata.normalize('NFKD', partFilename).encode('ascii', 'ignore'))
    temp = temp[2:]
    file_name = temp[:-1]

    #数据处理
    soup = BeautifulSoup(open(filename, encoding="utf-8", errors='ignore'))

    # content=reserve_chinese(soup.find('div',class_='content').text)

    # 实体提取 1.标题
    try:
        title = soup.find('h1').text.replace("\n", "")
    except:
        print(time.asctime(time.localtime(time.time())) + " " + file_name + ": 没有标题")
        return

    # 实体提取 2.数学公式
    contentForMath = re.sub(r"<mstyle[\s\S]*?>", "", open(filename, encoding="utf-8", errors='ignore').read())
    contentForMath = re.sub(r"</mstyle>", "", contentForMath)
    formularTrees = MathExtractor.parse_from_xml(contentForMath, 1)



    formulars=soup.find_all('math')

    formular_tag=0
    if len(formulars) == 0 and len(formularTrees) == 0:
        formular_tag=1
        print(time.asctime(time.localtime(time.time()))+" "+file_name+": 没有数学公式")
    elif len(formulars) != len(formularTrees):
        formular_tag=1
        print(time.asctime(time.localtime(time.time()))+" "+file_name+": 数学公式数目不一样")

    if formular_tag == 0:
        formularNum=0
        for i in formularTrees.keys():
            f = formulars[i].text
            # handle.add_relationship("Title", title, "EXIST", "Formular", f, id(formularTrees[i]))
            formularSubTreeAttr=getFormularAttr(getsubtreeBySLT(formularTrees[i]))
            # FDSFormularSymbols = tree2FDS(formularTrees[i])
            try:
                handle.add_property("Formular", f, "formularSubTreeAttr", formularSubTreeAttr)
                # handle.add_property("Formular", f, "FormularPickle", pickle.dump(formularTrees[i],open("temp.pkl",'wb')))
            except:
                print(time.asctime(time.localtime(time.time())) + " " + file_name + ": 存储公式FDS属性出错")
            # try:
                # handle.add_property("Formular", f, "formularNum", file_name+":"+str(formularNum))
            # except:
            #     print(time.asctime(time.localtime(time.time())) + " " + str(formularNum) + ": 存储公式文件名属性出错")

            formularNum+=1
    # 实体提取 3.作者
    try:
        authors = soup.find('h2').text
        authorsLists = authors[1:len(authors) - 1].split('\n')
        for a in authorsLists:
            handle.add_relationship("Title", title, "WRITE", "Author", a)
    except:
        print(time.asctime(time.localtime(time.time())) + " " + file_name + ": 没有作者或存储出错")
    # 实体提取 4.期刊
    try:
        journal = soup.find('div', class_='tips').text.replace("\n", "")
        journal = journal[0:journal.index(' ')]
        handle.add_relationship("Title", title, "PUBLISH", "Journal", journal)
    except:
        print(time.asctime(time.localtime(time.time())) + " " + file_name + ": 没有期刊或存储出错")


    # 实体提取 5.关键字
    try:
        keyWord = soup.find('div', attrs={"id": "a_keywords"}).find('p').text
        keyWordLists = keyWord[1:len(keyWord) - 1].split('\n')
        for k in keyWordLists:
            handle.add_relationship("Title", title, "HAVE", "Keyword", k)
    except Exception as e:
        print(time.asctime(time.localtime(time.time()))+" "+file_name+": 没有关键字或存储出错")

    # 实体提取 6.参考文献
    try:
        references=[]
        s=soup.find_all('div',class_='reference anchor-tag')
        if len(s)!=0:
            references=s[0].find_all('a')
        referencesList=[]
        # 实体提取 7.引用 作者，名称，期刊
        for reference in references:
            # referenceList = []
            tags = reference.text.replace("\n", "").split('.')
            r=reference.text.replace("\n", "")
            # referenceList.append(r)
            # referenceList.append(tags[0][tags[0].index(' ')+1:].split(','))
            # referenceList.append(tags[1])
            # referenceList.append(tags[2][:tags[2].index(',') if ',' in tags[2] else len(tags[2])-1])
            referencesList.append(r)
        for reference in referencesList:
            handle.add_relationship("Title", title, "CITE", "Reference", reference)
    except:
        print(time.asctime(time.localtime(time.time())) + " " + file_name + ": 没有引用或存储出错")


def addFormularSubTreeAttr(filename,handle):
    # filename
    partFilename = filename.split('/')[-1]
    partFilename = os.path.splitext(partFilename)[0]
    temp = str(unicodedata.normalize('NFKD', partFilename).encode('ascii', 'ignore'))
    temp = temp[2:]
    file_name = temp[:-1]

    # 数据处理
    soup = BeautifulSoup(open(filename, encoding="utf-8", errors='ignore'))

    contentForMath = re.sub(r"<mstyle[\s\S]*?>", "", open(filename, encoding="utf-8", errors='ignore').read())
    contentForMath = re.sub(r"</mstyle>", "", contentForMath)
    formularTrees = MathExtractor.parse_from_xml(contentForMath, 1)

    formulars = soup.find_all('math')

    formular_tag = 0
    if len(formulars) == 0 and len(formularTrees) == 0:
        formular_tag = 1
        print(time.asctime(time.localtime(time.time())) + " " + file_name + ": 没有数学公式")
    elif len(formulars) != len(formularTrees):
        formular_tag = 1
        print(time.asctime(time.localtime(time.time())) + " " + file_name + ": 数学公式数目不一样")

    if formular_tag == 0:
        formularNum = 0
        for i in formularTrees.keys():
            f = formulars[i].text
            # handle.add_relationship("Title", title, "EXIST", "Formular", f, id(formularTrees[i]))
            formularSubTreeAttr = getFormularAttr(getsubtreeBySLT(formularTrees[i]))
            subarr=list(formularSubTreeAttr.values())
            try:
                handle.add_property("Formular", f, "formularSubTreeAttr", subarr)
                # handle.add_property("Formular", f, "FormularPickle", pickle.dump(formularTrees[i],open("temp.pkl",'wb')))
            except:
                print(time.asctime(time.localtime(time.time())) + " " + file_name + ": 存储公式子式属性出错")
            # try:
            # handle.add_property("Formular", f, "formularNum", file_name+":"+str(formularNum))
            # except:
            #     print(time.asctime(time.localtime(time.time())) + " " + str(formularNum) + ": 存储公式文件名属性出错")
            print(time.asctime(time.localtime(time.time())) + " " + file_name + "已存储子式")
            formularNum += 1



if __name__ == '__main__':
    # extrSingleFile("中文数据集/J中学数学教学/J中学数学教学201901-9-12.html")
    extr()
    # extrPackage()
