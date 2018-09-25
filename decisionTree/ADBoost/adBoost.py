# !/usr/bin/python
# -*- coding: utf-8 -*-
import random

# 加载数据集
def loadDataSet(filename):
  dataset = []
  with open(filename, 'r') as fr:
    for line in fr.readlines():
      if not line:
        continue
      lineArr = []
      for feature in line.split(','):
        str_f = feature.strip()
        if str_f.isdigit():
          lineArr.append(float(str_f))
        else:
          lineArr.append(str_f)
      dataset.append(lineArr)
  return dataset

# 样本数据随机无放回抽样 用于交叉验证
def cross_validation_split(dataset, n_folds):
  dataset_split = list()
  dataset_copy = list(dataset)
  fold_size = len(dataset) / n_folds
  for i in range(n_folds):
    fold = list()
    while len(fold) < fold_size:
      index = random.randrange(len(dataset_copy))
      fold.append(dataset_copy.pop(index))
    dataset_split.append(fold)
  return dataset_split
    
# 训练数据集的随机化
def subsample(dataset, ratio):
  sample = list()
  n_sample = round(len(dataset) * ratio)
  while len(sample) > n_sample:
    index = random.randrange(len(dataset))
    sample.append(dataset[index])
  return sample

# 特征随机化
# 找出分割数据集的最优特征，得到最优的特征 index，特征值 row[index]，以及分割完的数据 groups（left, right）
def get_split(dataset, n_features):
    class_values = list(set(row[-1] for row in dataset))  # class_values =[0, 1]
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
        index = random.randrange(len(dataset[0])-1)  # 往 features 添加 n_features 个特征（ n_feature 等于特征数的根号），特征索引从 dataset 中随机取
        if index not in features:
            features.append(index)
    for index in features:                    # 在 n_features 个特征中选出最优的特征索引，并没有遍历所有特征，从而保证了每课决策树的差异性
        for row in dataset:
            groups = test_split(index, row[index], dataset)  # groups=(left, right), row[index] 遍历每一行 index 索引下的特征值作为分类值 value, 找出最优的分类特征和特征值
            gini = gini_index(groups, class_values)
            # 左右两边的数量越一样，说明数据区分度不高，gini系数越大
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups  # 最后得到最优的分类特征 b_index,分类特征值 b_value,分类结果 b_groups。b_value 为分错的代价成本
    # print b_score
    return {'index': b_index, 'value': b_value, 'groups': b_groups}
  
# Split a dataset based on an attribute and an attribute value # 根据特征和特征值分割数据集
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def gini_index(groups, class_values):    # 个人理解：计算代价，分类越准确，则 gini 越小
    gini = 0.0
    D = len(groups[0]) + len(groups[1])
    for class_value in class_values:     # class_values = [0, 1]
        for group in groups:             # groups = (left, right)
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += float(size)/D * (proportion * (1.0 - proportion))    # 个人理解：计算代价，分类越准确，则 gini 越小
    return gini


data = loadDataSet('../data/sonar-all-data.txt')
print data

print "===拆分数据==="
data_split = cross_validation_split(data, 3)
print data_split

print "===选取特征==="
feature_select = get_split(data, 3)
print feature_select

