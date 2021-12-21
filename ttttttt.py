#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/16 17:25
# @Author  : LiXin
# @File    : ttttttt.py
# @Describe:
import numpy as np
import tensorflow as tf
import os
from transformers import BertTokenizer,BertModel,AutoTokenizer,AutoModel
import jieba
from gensim.models import word2vec
# from gensim.models.word2vec import KeyedVectors
from simhash import Simhash
from neo4jUtil import Neo4j_handle
import sys
import pickle
import time
import re
import codecs
import jieba.posseg as pseg
from gensim import corpora
from collections import Counter
from gensim.summarization import bm25
from bs4 import BeautifulSoup
import numpy as np
from ast import literal_eval
import unicodedata
from utils import MemTit
from TangentS.math_tan.math_extractor import MathExtractor
from MathUtil import getsubtreeBySLT


def findFormular(filename,num):

    # 数据处理
    soup = BeautifulSoup(open(filename, encoding="utf-8", errors='ignore'))

    contentForMath = re.sub(r"<mstyle[\s\S]*?>", "", open(filename, encoding="utf-8", errors='ignore').read())
    contentForMath = re.sub(r"</mstyle>", "", contentForMath)
    formularTrees = MathExtractor.parse_from_xml(contentForMath, 1)

    formulars = soup.find_all('math')

    formularNum = 0
    for i in formularTrees.keys():
        if formularNum==num:
            f = formulars[i].text
            print("111")
        formularNum += 1

if __name__ == '__main__':
    findFormular("中文数据集/J中学数学杂志/J中学数学杂志2019-03-60-62.html",0)

sys.exit(0)

a=[1]
b=[1]
s=min(a[0],b[0])/max(a[0],b[0])
print(s)

filename="query/query10.html"
c = open(filename, encoding="utf-8", errors='ignore').read()
contentForMath = re.sub(r"<mstyle[\s\S]*?>", "", c)
contentForMath = re.sub(r"</mstyle>", "", contentForMath)
# 获取公式结构树
formularTree = MathExtractor.parse_from_xml(contentForMath, 1)
getsubtreeBySLT(formularTree)


dict={'a':1,"b":2}


a=(1,2,3,4)
b=(4,3,2,1)
print(sum(min(a[i],b[i]) for i in range(len(a))))

file="query0.html"
if file in {"query0.html","query1.html","query2.html","query4.html","query5.html","query9.html"}:print("1")



print(MemTit("基于边缘映射抽样与多维尺度压缩的紧凑图像哈希算法","视频监控中私自揽客违法行为检测"))
print(MemTit("基于边缘映射抽样与多维尺度压缩的紧凑图像哈希算法","基于小波变换的分形图像编码压缩算法"))
print(MemTit("基于边缘映射抽样与多维尺度压缩的紧凑图像哈希算法","基于深度多监督哈希的快速图像检索"))



s='J2019-09-9-15:156'
for fileDir in os.listdir("中文数据集"):
    dirChild = "中文数据集" + "/" + fileDir
    for file in os.listdir(dirChild):
        filename = dirChild + "/" + file
        partFilename = filename.split('/')[-1]
        partFilename = os.path.splitext(partFilename)[0]
        temp = str(unicodedata.normalize('NFKD', partFilename).encode('ascii', 'ignore'))
        temp = temp[2:]
        file_name = temp[:-1]
        if file_name == s[:s.index(":")]:
            print("11")


sys.exit(0)

D={1:'2'}
s = sorted(D.items(), key=lambda x: x[1], reverse=True)
print("1")


d = os.path.abspath(os.path.dirname(__file__))
temp_address = os.path.join(d + '/../code2/slt_model.wv.vectors.npy')
a=np.load(temp_address, allow_pickle=True).item()
print("111")



read_dictionary = np.load('aaaaaa.npy', allow_pickle=True).item()
print(read_dictionary)

sys.exit(0)


dic={}
dic['a']='b'
np.save("aaaaaa.npy",dic)




sys.exit(0)


class BM25(object):
  def __init__(self,docs):
    self.docs = docs   # 传入的docs要求是已经分好词的list
    self.doc_num = len(docs) # 文档数
    self.vocab = set([word for doc in self.docs for word in doc]) # 文档中所包含的所有词语
    self.avgdl = sum([len(doc) + 0.0 for doc in docs]) / self.doc_num # 所有文档的平均长度
    self.k1 = 1.5
    self.b = 0.75

  def idf(self,word):
    if word not in self.vocab:
      word_idf = 0
    else:
      qn = {}
      for doc in self.docs:
        if word in doc:
          if word in qn:
            qn[word] += 1
          else:
            qn[word] = 1
        else:
          continue
      word_idf = np.log((self.doc_num - qn[word] + 0.5) / (qn[word] + 0.5))
    return word_idf

  def score(self,word):
    score_list = []
    for index,doc in enumerate(self.docs):
      word_count = Counter(doc)
      if word in word_count.keys():
        f = (word_count[word]+0.0) / len(doc)
      else:
        f = 0.0
      r_score = (f*(self.k1+1)) / (f+self.k1*(1-self.b+self.b*len(doc)/self.avgdl))
      score_list.append(self.idf(word) * r_score)
    return score_list

  def score_all(self,sequence):
    sum_score = []
    for word in sequence:
      sum_score.append(self.score(word))
    sim = np.sum(sum_score,axis=0)
    return sim


stop_words = 'stop_words.txt'
stopwords = codecs.open(stop_words,'r',encoding='utf8').readlines()
stopwords = [ w.strip() for w in stopwords ]
stop_flag = ['x', 'c', 'u','d', 'p', 't', 'uj', 'm', 'f', 'r']
def tokenization(filename):
    result = []
    soup=BeautifulSoup(open(filename, encoding="utf-8", errors='ignore'))
    content=soup.find('div', class_='content').text
    words = pseg.cut(content)
    for word, flag in words:
        if flag not in stop_flag and word not in stopwords:
            result.append(word)
    return result

corpus = []
filenames = []
for root, dirs, files in os.walk('中文数据集/J中国卫生统计'):
    for f in files:
        corpus.append(tokenization(root+"/"+f))
        filenames.append(f)
dictionary = corpora.Dictionary(corpus)
print(len(dictionary))

bm = BM25(corpus)
query=['卵巢癌','患者','发病','危险','因素','术后康复','效果','影响','因素','分析','中国','卫生','统计']
score = bm.score_all(query)
print(score)
print(max(score))
print(np.where(score==np.max(score)))

sys.exit(0)

doc_vectors = [dictionary.doc2bow(text) for text in corpus]
vec1 = doc_vectors[0]
# print("1")
vec1_sorted = sorted(vec1, key=lambda x:x[1], reverse=True)
print(len(vec1_sorted))
for term, freq in vec1_sorted:
    print(dictionary[term])

bm25Model=bm25.BM25(corpus)
average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())

query_str="深度学习"
query=[]
for word in query_str.strip().split():
    query.append(word)
# scores=bm25Model.get_scores(query,average_idf)
# print(scores)
# scores.sorted(reverse=True)
# idx=scores.index(max(scores))




# print(time.asctime(time.localtime(time.time()))+".")

# pickle.dump("a",open("temp.pkl",'wb'))
# print(pickle.load(open("temp.pkl","rb")))

# print(str([1,2,3]))

# handle=Neo4j_handle()
# handle.add_property("paper","aaaa","title","bbbb")


# tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
# model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')



content=['分析2010-2017年八师结直肠癌的发病、死亡及疾病负担情况,为该地区结直肠癌的防控提供依据。方法 根据2010-2017年八师肿瘤登记数据,计算结直肠癌发病率、死亡率、年度变化百分比(APC)、伤残调整寿命年(disability adjusted life year,DALY)等指标。结果 2010-2017年八师居民结直肠癌发病率、死亡率为23.50/10万和13.90/10万,男性高于女性(P<0.05)。结直肠癌发病率、死亡率呈上升趋势,APC分别为9.88%、15.35%。结直肠癌每千人YLL、YLD、DALY分别为1.36、0.07、1.43,男性高于女性(P<0.05)。男、女DALY高峰值分别在65～69岁和60～64岁组。结论 2010-2017年八师居民结直肠癌发病和死亡水平较高,早死是疾病负担主要来源,应重点对45岁以上男性及60岁以上人群开展早诊早治,降低结直肠癌流行水平。',
   '利用美国监测、流行病学和最终结果数据库SEER(surveillance,epidemiology,and end results)对绝经后早期宫颈癌患者的预后因素展开研究,以指导临床预后判断和治疗决策。方法 从SEER数据库中提取的变量有患者人口学特征、肿瘤特征、治疗方式和生存结果。采用Kaplan-Meier法构建生存曲线,并采用log-rank检验评估曲线之间的统计学差异。结果 年龄增加(OS:HR=2.750,95%CI:2.440～3.098,P<0.05)、无伴侣(OS:HR=1.154,95%CI:1.016～1.311,P<0.05)、肿瘤组织学分级低分化(OS:HR=1.290,95%CI:1.073～1.552,P<0.05)、腺癌(OS:HR=1.298,95%CI:1.148～1.468,P<0.05)、不接受手术治疗(OS:HR=1.348,95%CI:1.186～1.533,P<0.05)是降低绝经后妇女早期宫颈癌总生存率的独立预后因素。城乡因素未被纳入CSS的Cox回归模型。肿瘤分期IA1期(CSS:HR=0.276,95%CI:0.121～0.629,P<0.05;OS:HR=0.396,95%CI:0.280～0.562,P<0.05)是提高绝经后妇女早期宫颈癌特异性生存率的独立预后因素。结论 年龄增加、组织学分级低分化、腺癌、不接受手术治疗会增加绝经后妇女早期宫颈癌患者的死亡风险,有伴侣、Ⅰ期肿瘤对于绝经后早期宫颈癌患者总体生存状况来说具有一定保护作用。',
   '我国全人口近视发病率为33%,是世界平均水平的1.5倍[1]。而学生处于生长发育期,眼球的调节力很强,如长期处于高度紧张的调节状态,晶状体的曲度逐渐增加,球壁伸长,眼轴拉长,容易引发近视[2]。学生既已成为近视的高危人群[3]。2005年我国学生近视患病率达60%,居世界第2位,患病人数居世界第1位[4]。近视不仅是我国现代校园三大公共卫生问题之一[5],更是全世界范围内的公共卫生问题[6]。近视具有叠加、渐进和不可逆的特点,高度近视可以引发近视性黄斑变性、原发性开角型青光眼和视网膜脱落等并发症[5],将严重影响学生的健康及未来的生活,近视防控工作越早开展效果越好,小学生应成为预防工作的重点人群。',
   '上海统计年鉴数据显示,2016年上海市恶性肿瘤死亡专率为262.54/10万,占死亡总数的30.79%,仅次于循环系统疾病(343.12/10万,40.24%)[1],是上海市居民第二大死因。肿瘤患者病情重、治疗时间长、并发症多、用药量大,患者经济负担重。研究证实恶性肿瘤患者约70%费用集中在临终前半年[2],既往研究结果显示年龄[3,4]、性别[5,6]、住院天数[7,8]、是否手术[9]、距离死亡时间[10]、临终前住院机构[4]、癌种[5]等均是恶性肿瘤患者住院费用的影响因素,然而目前国内鲜有关于临终期恶性肿瘤患者住院费用影响因素的研究。因此本研究利用上海市2016年在医疗机构内死于恶性肿瘤患者的就诊信息,深入研究临终期住院费用的影响因素,为相关部门合理配置医疗资源和合理控制住院费用提供一定的依据,也为我国其他地区研究临终期住院费用的影响因素提供参考。']
# print(tokenizer.encode(s))
seg=[jieba.lcut(text) for text in content]


model=word2vec.Word2Vec(seg,size=100,min_count=1)
print(model.wv.index2word)
# print(model.wv.get_vector('分析'))

class a:
    def b(self,m,n):
        print(m+n)
a().b(1,2)


a=np.empty([300,])
b=np.empty([1,5])
# print(b.dot(a))


print(np.array([2,2])/2)


def getDCG(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)

def getNDCG(rank_list, pos_items):
    relevance = np.ones_like(pos_items)
    it2rel = {it: r for it, r in zip(pos_items, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)

    idcg = getDCG(relevance)

    dcg = getDCG(rank_scores)

    if dcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg

l1 = [1, 4, 5]
l2 = [1, 2, 3]
a = getNDCG(l1, l2)
print(a)

dict={1:1}
print(list(dict.values()))

