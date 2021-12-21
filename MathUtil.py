#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/26 16:43
# @Author  : LiXin
# @File    : MathUtil.py
# @Describe:
import numpy as np

#[符号，序号，层次，操作符，位置flag，占比]


#层次隶属度
def MemLevel(qelev, fslev):
    level_value = np.exp(-abs(qelev-fslev))
    if level_value<0 or level_value>1:
        level_value = 1/level_value
    return level_value

# 用于计算数学公式每个符号占比的隶属度
def MemRatio(qeratio, fsratio):
    u = 1
    v = 1
    ratio = 1/(1+u*abs(qeratio-fsratio)**v)
    return ratio


# 用于计算数学公式每个符号操作符的隶属度
def MemOpe(qeope, fsope):
    if qeope == fsope:
        return 1
    else:
        return 0.5

# 用于计算数学公式每个符号标志位的隶属度
def MemFlag(qeflag, fsflag):
    if qeflag == fsflag:
        return 1
    else:
        return 0

#数量隶属度
def MemNum(qenum, fsnum):
    deta = 1
    num = np.exp(-((qenum-fsnum)/deta) ** 2)
    return num

#得到符号集的长度
def getlen(SymbolArr):
    loc = {}
    maxl=0
    for symbol in SymbolArr:
        if symbol[4] not in loc.keys():loc[symbol[4]]=1
        else: loc[symbol[4]]+=1
        maxl=maxl if maxl>loc[symbol[4]] else loc[symbol[4]]
    return maxl

def tree2FDS(formularTree):
    pre_op = []
    root = formularTree.root
    scanTree(root, pre_op, 1, 0, 0)
    FDS=[]
    for item in pre_op[::-1]:
        item.append(1/len(pre_op))
    for item in pre_op[::-1]:
        FDS.append(item)
        # print(item)
    # print("--------------")
    return FDS

def getSim_HFS(queryArray,FDSArray,f_id=None):
    HFS5Tuple=getHFS5Tuple(queryArray,FDSArray,f_id)
    f_tuple_temp=np.array([0.0,0.0,0.0,0.0,0.0])
    for temp in HFS5Tuple:
        f_tuple_temp +=np.abs(np.array([1.0,1.0,1.0,1.0,1.0])-np.array(temp[2]))

    f_tuple_temp=np.array(f_tuple_temp)/len(HFS5Tuple)
    sim = 1-np.sum(f_tuple_temp)/5
    return sim

def getSim_DHFS(queryArray,FDSArray,f_id=None):
    DHFS5Tuple=getDHFS5Tuple2(queryArray,FDSArray,f_id)
    f_tuple_temp=np.array([(0.0,0.0),(0.0,0.0),(0.0,0.0),(0.0,0.0),(0.0,0.0)])
    for temp in DHFS5Tuple:
        f_tuple_temp +=np.array(temp[2])
    f_tuple_temp = np.array(f_tuple_temp) / len(DHFS5Tuple)
    a=np.subtract(f_tuple_temp[:,0],f_tuple_temp[:,1])
    sim=np.sum(a,axis=0)/5
    sim=pow(sim,1/2)
    return sim

def getHFS5Tuple(queryArray,FDSArray,f_id=None):
    array=[]
    for q_s in queryArray:
        symbolArray = []
        maxTuple=0
        for d_s in FDSArray:
            # 层次 位置 数量 占比 操作符
            HFS5Tuple=[MemLevel(q_s[2],d_s[2]),MemFlag(q_s[4],d_s[4]),MemNum(q_s[1],d_s[1]),MemRatio(q_s[5],d_s[5]),MemOpe(q_s[3],d_s[3])]
            avg=sum(HFS5Tuple)/5
            if avg>maxTuple:
                maxTuple=avg
                value=[q_s[0],f_id,HFS5Tuple]
                symbolArray=value
        array.append(symbolArray)
    return array

def getDHFS5Tuple(queryArray,FDSArray,f_id=None):
    array = []
    for q_s in queryArray:
        symbolArray = []
        maxTuple = 0
        for d_s in FDSArray:
            # 层次 位置 数量 占比 操作符
            HFS5Tuple = [MemLevel(q_s[2], d_s[2]), MemFlag(q_s[4], d_s[4]), MemNum(q_s[1], d_s[1]),
                         MemRatio(q_s[5], d_s[5]), MemOpe(q_s[3], d_s[3])]
            # avgMem = sum(HFS5Tuple) / 5
            NonHFS5Tuple=[1-HFS5Tuple[0],1-HFS5Tuple[1],1-HFS5Tuple[2],1-HFS5Tuple[3],1-HFS5Tuple[4]]
            # avgNonMem=sum(NonHFS5Tuple)/5
            s1=sorted(HFS5Tuple,reverse=True)
            s2=sorted(NonHFS5Tuple,reverse=True)
            avg=sum(abs(s1[i]-s2[i]) for i in range(5))/5
            DHFSArray=[(HFS5Tuple[i],NonHFS5Tuple[i]) for i in range(5)]
            if avg > maxTuple:
                maxTuple = avg
                value = [q_s[0], f_id, DHFSArray]
                symbolArray = value
        array.append(symbolArray)
    return array

def getDHFS5Tuple2(queryArray,FDSArray,f_id=None):
    array = []
    for q_s in queryArray:
        symbolArray = []
        maxTuple = 0
        for d_s in FDSArray:
            # 层次 位置 数量 占比 操作符
            HFS5Tuple = [MemLevel(q_s[2], d_s[2]), MemFlag(q_s[4], d_s[4]), MemNum(q_s[1], d_s[1]),
                         MemRatio(q_s[5], d_s[5]), MemOpe(q_s[3], d_s[3])]
            # avgMem = sum(HFS5Tuple) / 5
            NonHFS5Tuple=[1-HFS5Tuple[0],1-HFS5Tuple[1],1-HFS5Tuple[2],1-HFS5Tuple[3],1-HFS5Tuple[4]]
            # avgNonMem=sum(NonHFS5Tuple)/5
            s1=sorted(HFS5Tuple,reverse=True)
            s2=sorted(NonHFS5Tuple,reverse=True)
            avg=sum(pow(abs(s1[i]-s2[i]),2) for i in range(5))/5
            DHFSArray=[(HFS5Tuple[i],NonHFS5Tuple[i]) for i in range(5)]
            if avg > maxTuple:
                maxTuple = avg
                value = [q_s[0], f_id, DHFSArray]
                symbolArray = value
        array.append(symbolArray)
    return array


def getHFS(FDSArray):
    all_subf=getsubf(FDSArray)
    #计算HFS值，层次、长度（判断同一位置最长）、运算符个数len
    #先计算整个公式的长度及运算符个数
    #长度
    f_l=getlen(FDSArray)
    #运算符个数
    f_n=len(FDSArray)

    HFSArray=[]
    #计算子式长度及运算符个数
    #长度
    for level in all_subf.keys():
        sf_l=getlen(all_subf[level])
        sf_n=len(all_subf[level])
        Ul=sf_l/f_l
        Un=sf_n/f_n
        Ulevel=1/pow(level+1,1/3)
        subHFS=[Ul,Un,Ulevel]
        HFSArray.append(subHFS)
    return HFSArray







def getsubf(FDSArray):
    all_subf={}
    for symbol in FDSArray:
        if symbol[2] not in all_subf.keys():
            subf=[symbol]
            all_subf[symbol[2]]=subf
        else:
            all_subf[symbol[2]].append(symbol)
    return all_subf

globol_index = 0


def scanTree(root, op, index, level, flag):
    global globol_index
    globol_index = index
    if root.next is not None:
        scanTree(root.next, op, globol_index + 1, level, flag)
    if root.above is not None:
        scanTree(root.above, op, globol_index + 1, level + 1, 2)
    if root.below is not None:
        scanTree(root.below, op, globol_index + 1, level + 1, 4)
    if root.pre_above is not None:
        scanTree(root.pre_above, op, globol_index + 1, level + 1, 7)
    if root.pre_below is not None:
        scanTree(root.pre_below, op, globol_index + 1, level + 1, 8)
    if root.within is not None:
        scanTree(root.within, op, globol_index + 1, level + 1, 6)
    if root.under is not None:
        scanTree(root.under, op, globol_index + 1, level + 1, 5)
    if root.over is not None:
        scanTree(root.over, op, globol_index + 1, level + 1, 1)

    array = [root.tag,index, level, 1 if 'N' not in root.tag and 'V' not in root.tag else 0, flag]
    op.append(array)

def getDCG(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)

def getNDCG(scores,k):
    if scores is None :return 0
    s=list(scores.values())
    dcg = getDCG(np.array(s[:k]))
    s.sort(reverse=True)
    idcg = getDCG(np.array(s[:k]))

    if dcg == 0.0:
        return 0.0
    ndcg = dcg / idcg
    return ndcg

def getNDCG2(scores,k):
    if scores is None :return 0
    s=list(scores.values())
    s.sort(reverse=True)
    dcg = getDCG(np.array(s[:k]))
    ilist=[1]*k
    idcg = getDCG(np.array(ilist))
    if dcg == 0.0:
        return 0.0
    ndcg = dcg / idcg
    return ndcg

def getAP(scores,k,ap_value=0.6):
    ap=0
    i=1
    j=0
    for item in scores.values():
        j += 1
        if item>ap_value:
            ap+=i/j
            i+=1
            if i > k: break
    return ap/k
def getAP2(scores,k):
    a=list(scores.values())
    a.sort(reverse=True)
    ap = sum(a[:k])
    return ap/k

def getsubtreeBySLT(tree):
    subtrees=[]
    getsubtree(tree.root,subtrees)
    return subtrees

def getsubtree(tree,subtress):
    if tree.tag is None:
        return
    if tree.next is not None:
        getsubtree(tree.next,subtress)
    if tree.within is not None:
        getsubtree(tree.within,subtress)
    if tree.above is not None:
        getsubtree(tree.above,subtress)
    if tree.below is not None:
        getsubtree(tree.below,subtress)
    if tree.element is not None:
        getsubtree(tree.element,subtress)
    if tree.over is not None:
        getsubtree(tree.over,subtress)
    if tree.pre_above is not None:
        getsubtree(tree.pre_above,subtress)
    if tree.pre_below is not None:
        getsubtree(tree.pre_below,subtress)
    if tree.under is not None:
        getsubtree(tree.under,subtress)

    subtress.append(tree)

def getcontent(tree,s):
    if tree.tag is None:
        return
    if tree.next is not None:
        getcontent(tree.next, s)
    if tree.within is not None:
        getcontent(tree.within, s)
    if tree.above is not None:
        getcontent(tree.above, s)
    if tree.below is not None:
        getcontent(tree.below, s)
    if tree.element is not None:
        getcontent(tree.element, s)
    if tree.over is not None:
        getcontent(tree.over, s)
    if tree.pre_above is not None:
        getcontent(tree.pre_above, s)
    if tree.pre_below is not None:
        getcontent(tree.pre_below, s)
    if tree.under is not None:
        getcontent(tree.under, s)
    s.append(tree.tag)

def getSubTreelen(tree):
    if tree.tag is None:
        return 0
    if tree.next is not None:
        return 1+getSubTreelen(tree.next)
    if tree.within is not None:
        return 1+getSubTreelen(tree.within)
    if tree.above is not None:
        return 1+getSubTreelen(tree.above)
    if tree.below is not None:
        return 1+getSubTreelen(tree.below)
    if tree.element is not None:
        return 1+getSubTreelen(tree.element)
    if tree.over is not None:
        return 1+getSubTreelen(tree.over)
    if tree.pre_above is not None:
        return 1+getSubTreelen(tree.pre_above)
    if tree.pre_below is not None:
        return 1+getSubTreelen(tree.pre_below)
    if tree.under is not None:
        return 1+getSubTreelen(tree.under)
    return 1

def getlevel(tree):
    if tree.tag is None:
        return 0
    if tree.within is not None:
        return 1+getlevel(tree.within)
    if tree.above is not None:
        return 1+getlevel(tree.above)
    if tree.below is not None:
        return 1+getlevel(tree.below)
    if tree.element is not None:
        return 1+getlevel(tree.element)
    if tree.over is not None:
        return 1+getlevel(tree.over)
    if tree.pre_above is not None:
        return 1+getlevel(tree.pre_above)
    if tree.pre_below is not None:
        return 1+getlevel(tree.pre_below)
    if tree.under is not None:
        return 1+getlevel(tree.under)
    return 1


def getFormularAttr(sub):
    sub_attr = {}
    for s in sub:
        c = []
        getcontent(s, c)
        temp = [getSubTreelen(s), getlevel(s), c, c]
        if [getSubTreelen(s), getlevel(s), c, c] in sub_attr.values(): continue
        sub_attr[s] = temp
    return sub_attr

def getEqu(arr1,arr2):
    cnt=0
    if len(arr1)>len(arr2):
        for item in arr1:
            if item in arr2:cnt+=1
    else:
        for item in arr2:
            if item in arr1:cnt+=1
    return cnt

def getNORcontent(arr1,arr2):
    #脱水
    n_arr1={}
    n_arr2={}
    maxn=0
    for item in arr1:
        if item[0] in n_arr1.keys():
            n_arr1[item[0]]+=1
        else:
            n_arr1[item[0]]=1
    for item in arr2:
        if item[0] in n_arr2.keys():
            n_arr2[item[0]]+=1
        else:
            n_arr2[item[0]]=1
    if len(n_arr1.keys())>len(n_arr2.keys()):
        for item in n_arr1.keys():
            if item in n_arr2.keys():
                a=min(n_arr1[item],n_arr2[item])/max(n_arr1[item],n_arr2[item])
                if a > maxn:maxn=a
    else:
        for item in n_arr2.keys():
            if item in n_arr1.keys():
                a=min(n_arr1[item],n_arr2[item])/max(n_arr1[item],n_arr2[item])
                if a > maxn:maxn=a
    return maxn


def getHFSsim(sub1,sub2,f_id=None):
    sum = 0
    for s1 in sub1:
        maxn = 0
        for s2 in sub2:
            memlen = min(s1[0], s2[0]) / max(s1[0], s2[0])
            memlevel = min(s1[1], s2[1]) / max(s1[1], s2[1])
            memcontent = getEqu(s1[2], s2[2]) / max(len(s1[2]), len(s2[2]))
            memNORcontent = getNORcontent(s1[3], s2[3])
            avg = (memlen + memlevel + memcontent + memNORcontent) / 4
            if avg > maxn: maxn = avg
        sum += maxn
    sim = sum / len(sub1)
    # print(sim)
    return sim