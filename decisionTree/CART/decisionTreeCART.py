# !/usr/bin/python
# -*- coding: utf-8 -*-

"""
  CART算法原理：
  CART是采用基尼系数来选择划分属性
  基尼系数越小，数据集的纯度越高
"""
import math
import argparse
import pygraphviz as pyg
import numpy as np

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

class DecisionTree:
  # def __init__(self, model='ID3'):
  #   if model == 'ID3' or model == 'C4.5' or model == 'CART':
  #     self._model = model
  #   else:
  #     raise Exception('model should be ID3 or C4.5 or CART')
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
    child = self.splitByCart(root)
    print "child: ", child
    for node in child:
        print "node.sample===", node.sample

    root.child = child
    if len(child) != 0:
        for i in child:
            self.build(i)

  def _caclGini(self, node):
    # 计算完整数据集的基尼系数
    num = len(node.sample)
    labelCount = {}
    for i in range(num):
      currentLabel = self.matrix[i][-1]
      print currentLabel
      labelCount[currentLabel] = labelCount.get(currentLabel, 0) + 1
    print "labelCount: ", labelCount
    GINI = 1.0
    for key in labelCount:
      prob = float(labelCount[key]) / num
      GINI -= prob ** 2
    print "全数据集Gini系数: ", GINI
    return GINI

  def splitByCart(self, node):
    """
      1、featureNum:特征个数
      2、baseGini: 原始数据集的基尼系数
      3、newGini: 按照某个特征分割数据集之后的基尼系数
      4、infoGini: 基尼指数增益
      5、maxGini: 最大的基尼系数增益
      6、bestFeatureIndex：基尼指数增益最大时，所选择的分割特征的下标
    """

    # 最大增益字典
    gain_max_dict = {}
    res = []
    if len(node.attribute) == 0:
        return res
    feature = node.attribute
    rowNum = node.sample
    print "feature count: ", feature
    baseGini = self._caclGini(node)
    print "baseGini: ", baseGini
    bestGini = 0.0
    bestFeatureIndex = -1

    # 遍历特征
    for f in feature:
        print "node sample: ", node.sample
        print "feature: ", f
        d = self.classify(node.sample, f)
        print "特征字典的长度: ", len(d)
        # for i in range(len(d)):
        g = self._caclSubGini(d)
        print "max gini: ", g
        # 为何要用全数据集基尼指数减去每个feature的基尼指数
        infoGini = baseGini - g
        print "infoGini: ", infoGini
        if infoGini > bestGini:
            bestGini = infoGini
            bestFeatureIndex = f
            gain_max_dict = d
    print "best gini: ", bestGini
    used_attr = node.attribute[:]
    print "bestFeatureIndex: ", bestFeatureIndex
    used_attr.remove(bestFeatureIndex) 

    print "gain_max_dict: ", gain_max_dict
    print "used_attr: ", used_attr
    for (k, v) in gain_max_dict.items():
        print "v is: ", v['count']
        res.append(Node(v['count'], used_attr, bestFeatureIndex, k))
    return res

  # 处理数据【获取特定feature的数据】   
  def classify(self, select_row, column):
    res = {}
    print 'column: ', column
    print 'select row: ', select_row
    labelCount = {}
    subLabel = {}
    for index in select_row:
        # 第index行、第column列
        key = self.matrix[index][column]
        # 第index行、最后一列
        label = self.matrix[index][-1]
        print 'classify key: ', key
        print 'classify label: ', label
        if key in res:
            res[key].append(index)
        else:
            res[key] = [index]
    print "===res===", res

    arr = []
    for key,values in res.items():
        arr.append(values)
    print "arr: ", arr
    index = 0
    for key,values in res.items():
        subLabelCount = {}
        # subLabelCount['cnt'] = 0
        for i in range(len(values)):
            currLabel = self.matrix[values[i]][-1]
            subLabelCount[currLabel] = subLabelCount.get(currLabel, 0) + 1
        print "sub label count: ", len(subLabelCount)
        counts = 0
        for k,v in subLabelCount.items():
            counts = counts + v
        print "counts: ", counts
        subLabelCount['cnt'] = counts
        subLabelCount['count'] = arr[index]
        res[key] = subLabelCount
        index = index + 1
    
    print 'res: ', res
    return res
  

  def _caclSubGini(self, res):
    # 计算完整数据集的基尼系数
    parentRow = len(self.matrix)
    maxGini = 0.0
    for key,values in res.items():
        p = float(res[key]['cnt']) / parentRow
        print "p is: ", p
        feaLen = len(res[key])
        print "feature len: ", feaLen
        print "values: ", values
        length = 0
        for key,results in values.items():
            if key != 'cnt' and key != 'count':
                print "results: ", results
                length += results
        print "length: ", length
        GINI = 1.0
        for k in values:
            if k != 'cnt' and k != 'count':
                prob = float(values[k]) / length
                print "prob== ", prob
                GINI -= prob ** 2
        
        print "GINI IS: ", GINI
        maxGini += p * GINI
    # print "max gini: ", maxGini
    return maxGini
        
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
  parser.add_argument('-f', '--file', dest='data', type=argparse.FileType('r'), default="../data.txt")
  args = parser.parse_args()
  matrix = []
  lines = args.data.readlines()
  for line in lines:
      print line
      matrix.append(line.split())
  print "matrix: ", matrix[0]
  CARTTree = DecisionTree(matrix)
#   CARTTree._caclGini(14, matrix)
#   CARTTree.splitByCart(matrix)
  # CARTTree.save("cart.png")


