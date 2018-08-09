# !/usr/bin/python
# -*- coding: utf-8 -*-

import math
import argparse
import pygraphviz as pyg
import numpy as np

"""
    1、计算类别信息熵
    2、计算每个属性的信息熵（即条件熵）
    3、计算信息增益
    4、计算属性分裂信息度量
    5、计算信息增益率（信息增益 / 属性分裂信息度量）
    ---选择分类属性节点：类别是纯的，定义为叶子节点；类别不纯，选择子节点重复1~5的过程
"""

class Node:
    """
        C45的决策树节点：包括
        1、节点采用哪些属性来分裂
        2、这个节点的字数包含哪些样例
        3、该节点本身是由父节点的哪个属性分裂来的，它的属性值是什么
    """
    def __init__(self, select_row, attribute, parent_a, value):
        self.sample = select_row
        self.attribute = attribute
        self.parent_attr = parent_a
        self.value = value
        self.child = []


class Tree:
    def __init__(self, data):
        self.matrix = data
        self.row = len(data)
        print 'row: ', self.row
        self.col = len(data[0])
        print 'col: ', self.col
        print list(range(self.row))
        print list(range(self.col - 1))
        self.root = Node(list(range(self.row)), list(range(self.col - 1)), self.col, 'root')
        self.build(self.root)
    
    def build(self, root):
        child = self.split(root)
        root.child = child
        if len(child) != 0:
            for i in child:
                self.build(i)

    def split(self, node):
        # 最大信息增益率
        gain_max = 0
        # 最大增益属性
        gain_max_attr = 0
        # 最大增益字典
        gain_max_dict = {}
        res = []
        if len(node.attribute) == 0:
            return res
        # [0, 1, 2, 3]
        for attr in node.attribute:
            # 对于每一个属性，计算其属性信息熵
            t = self.entropy(node.sample)
            if t == 0:
                return res
            d = self.classify(node.sample, attr)
            c = self.conditional_entropy(node.sample, d)
            c_e = (t - c[0]) / c[1]
            if c_e > gain_max:
                gain_max = c_e
                gain_max_attr = attr
                gain_max_dict = d
        used_attr = node.attribute[:]
        used_attr.remove(gain_max_attr)
        for (k, v) in gain_max_dict.items():
            res.append(Node(v, used_attr, gain_max_attr, k))
        return res


    def entropy(self, index_list):
        sample = {}
        for index in index_list:
            key = self.matrix[index][self.col - 1]
            print 'key: ', key
            # 统计no和yes的数量
            if key in sample:
                sample[key] += 1
            else:
                sample[key] = 1
        entropy_s = 0
        for k in sample:
            list_len = (float)(len(index_list))
#            print 'sample[k]: ', sample[k]
#            print 'index list len: ', list_len
#            print math.log(sample[k] / list_len, 2)
            entropy_s += -(sample[k] / list_len) * math.log(sample[k] / (float)(len(index_list)), 2)
        print '信息熵: ', entropy_s
        return entropy_s

    # 对每个属性进行归类
    def classify(self, select_row, column):
        res = {}
        print 'column: ', column
        for index in select_row:
            key = self.matrix[index][column]
            print 'classify key: ', key
            if key in res:
                res[key].append(index)
            else:
                res[key] = [index]
        print 'res: ', res
        return res

    # 计算每个属性的信息熵，即条件信息熵
    def conditional_entropy(self, select_row, d):
        c_e = 0
        # 所有的样本数量（也即多少行）
        total = (float)(len(select_row))
        c_info_measure = 0
        for k in d:
            # 求解条件信息熵
            c_e += (len(d[k]) / total) * self.entropy(d[k])
            # 求解属性分裂信息度量
#            print 'd[k] len: ', len(d[k])
#            print 'total: ', total
#            print '信息度量: ', math.log(len(d[k]) / total, 2)
            c_info_measure += -(len(d[k]) / total) * math.log(len(d[k]) / total, 2)

        return (c_e, c_info_measure)


    def save(self, filename):
        g = pyg.AGraph(strict=False, directed=True)
        g.add_node(self.root.value)
        self._save(g, self.root)
        g.layout(prog='dot')
        g.draw(filename)
        print("The file is save to %s." % filename)
    
    def _save(self, graph, root):
        if root.child:
            for node in root.child:
                graph.add_node(node.value)
                graph.add_edge(root.value, node.value)
                self._save(graph, node)
        else:
            graph.add_node(self.matrix[root.sample[0]], label=self.matrix[root.sample[0]][self.col - 1], shape="box")
            graph.add_edge(root.value, self.matrix[root.sample[0]])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='data', type=argparse.FileType('r'), default="data.txt")
    args = parser.parse_args()
    matrix = []
    lines = args.data.readlines()
    for line in lines:
        print line
        matrix.append(line.split())
    print matrix[0]
    C45Tree = Tree(matrix)
    C45Tree.save("data.png")

