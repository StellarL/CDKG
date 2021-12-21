#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/12/15 16:46
# @Author  : LiXin
# @File    : publishDate.py
# @Describe:
import requests
from neo4jUtil import Neo4j_handle


handler = Neo4j_handle()
titles=handler.get_all_Title()
requests.post("")
