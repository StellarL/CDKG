#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/25 15:03
# @Author  : LiXin
# @File    : ttttttt2.py
# @Describe:

from TangentS.math_tan.math_extractor import MathExtractor
from MathUtil import getsubtreeBySLT
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dict={"key1":"a","key2":"b"}
for item in dict:
    print("11")

import eventlet
import time
eventlet.monkey_patch()
#添加补丁
with eventlet.Timeout(2,False):
#最多执行时间为2s
    print(1)
    time.sleep(1.5)
    print(2)
    time.sleep(1)
    print(3)
    print(4)


ab=[1,2,3,4,1,1,1,2]
ab.sort()
df = pd.DataFrame(np.array(ab))

print(df.describe())
print(str(df.describe()))

plt.plot(np.array(ab))
plt.show()

dict={"k1":1,"k2":2}
score = sorted(dict.items(), key=lambda x: x[1], reverse=True)
print(score[0][1])
print("1")
