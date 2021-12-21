#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/15 21:13
# @Author  : LiXin
# @File    : data2Neo4j.py
# @Describe:

from py2neo import Graph,Node,Relationship,NodeMatcher,RelationshipMatcher
import time

class Neo4j_handle():

    graph=None
    nodeMatcher=None
    relationMatcher=None

    def __init__(self):
        # print("Neo4j Init ...")
        self.graph=Graph("http://127.0.0.1:7474")
        self.nodeMatcher=NodeMatcher(self.graph)
        self.relationMatcher=RelationshipMatcher(self.graph)
        self.graph.run("match (n) detach delete n")


    def add_node(self,label,name,id=None):
        # node = Node(label, name)
        # if self.graph.exists(node):pass
        # self.graph.create(node)
        if id is None:
            try:
                data = self.graph.run("MERGE(n:" + label + " {name:\"" + name + "\"})")
            except:
                data = self.graph.run("MERGE(n:" + label + " {name:\'" + name + "\'})")
        else:
            try:
                try:
                    data=self.graph.run("MERGE(n:"+label+" {name:\""+name+"\",id:"+str(id)+"})")
                except:
                    data = self.graph.run("MERGE(n:" + label + " {name:\'" + name + "\',id:" + str(id) + "})")
            except Exception as e:
                print("Neo4j创建节点失败"+e)

    def add_relationship(self,node1_label,node1_name,relationship_name,node2_label,node2_name,id=None):
        self.add_node(node1_label,node1_name)
        self.add_node(node2_label,node2_name,id)
        try:
            try:
                self.graph.run(
                    "MATCH (n1:"+node1_label+"{name:\""+node1_name+"\"}),(n2:"+node2_label+" {name:\""+node2_name+"\"}) MERGE (n1)-[r:"+relationship_name+"]->(n2) return n1,n2"
                )
            except:
                self.graph.run(
                    "MATCH (n1:" + node1_label + "{name:\"" + node1_name + "\"}),(n2:" + node2_label + " {name:\'" + node2_name + "\'}) MERGE (n1)-[r:" + relationship_name + "]->(n2) return n1,n2"
                )
        except Exception as e:
            print("Neo4j创建关系失败"+e)


    def add_property(self,node_label, node_name, property_name, property_value):
        self.add_node(node_label,node_name)
        try:
            self.graph.run("MATCH (n:" + node_label + "{name:\"" + node_name + "\"}) SET n."+property_name+"=\"" +str(property_value)+ "\" RETURN n")
        except:
            self.graph.run(
                "MATCH (n:" + node_label + "{name:\"" + node_name + "\"}) SET n." + property_name + "=\"" + str(
                    property_value) + "\" RETURN n")


    def delete_node(self,label,name):
        #detach删除节点及其所有的关系
        self.graph.run("MATCH(n:"+label+" {name:\""+name+"\"}) detach delete n")

    # def delete_reletionship(self, label):
    #     self.graph.run("MATCH(n:node1_label)-[r:]")


    def get_entity_info(self,name) -> list:
        """
        实体查询，查找该entity所有的直接关系
        :param name:
        :return:
        """
        data = self.graph.run(
            "match (source)-[rel]-(target)  where source.name = $name " +
            "return rel ", name=name).data()

        json_list = []
        for an in data:
            result = {}
            rel = an['rel']
            relation_type = list(rel.types())[0]
            start_name = rel.start_node['name']
            end_name = rel.end_node['name']
            result["source"] = start_name
            result['rel_type'] = relation_type
            result['target'] = end_name
            json_list.append(result)
        print(json_list)
        return json_list


