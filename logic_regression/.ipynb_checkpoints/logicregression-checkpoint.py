# !/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import *
import matplotlib.pyplot as plt

def sigmoid(inX):
  return 1.0/(1+exp(-inX))

def loadDataset():
  dataMat = []
  labelMat = []
  fr = open('testSet.txt')
  for line in fr.readlines():
    lineArr = line.strip().split()
    # 将x0设置为1.0  添加x1和x2数据
    dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
    # 结果预测值
    labelMat.append(int(lineArr[2]))
  return dataMat, labelMat

# 梯度下降
# 入参：
#   dataMatIn: 100 x 3
#   classLabels: 100 * 1
def gradAscent(dataMatIn, classLabels):
  dataMatrix = mat(dataMat)
  # 预测值做转置： 1 * 100
  labelMat = mat(classLabels).transpose()
  m, n = shape(dataMatrix)
  # 移动步长
  alpha = 0.001
  # 迭代次数
  maxCycles = 500
  weights = ones((n, 1))
  for k in range(maxCycles):
    h = sigmoid(dataMatrix*weights)
    error = (labelMat - h)
    # dataMatrix.transpose() * error  包含300次乘积
    weights = weights + alpha * dataMatrix.transpose() * error
  return weights

def plotBestFit(weights):
  dataMat, labelMat = loadDataset()
  dataArr = array(dataMat)
  # 训练数据的行数
  n = shape(dataArr)[0]
  xcord1 = []
  ycord1 = []
  xcord2 = []
  ycord2 = []
  for i in range(n):
    if (int(labelMat[i]) == 1):
      xcord1.append(dataArr[i, 1])
      ycord1.append(dataArr[i, 2])
    else:
      xcord2.append(dataArr[i, 1])
      ycord2.append(dataArr[i, 2])
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
  ax.scatter(xcord2, ycord2, s=30, c='green')
  x = arange(-3.0, 3.0, 0.1)
  y = (-weights[0]-weights[1]*x)/weights[2]
  ax.plot(x, y)
  plt.xlabel('X1')
  plt.ylabel('X2')
  plt.show()


# 随机梯度上升  一次随机选取一个样本点来更新回归系数
def randomGradAscent(dataMatrix, classLabels):
  m, n = shape(dataMatrix)
  alpha = 0.01
  weights = ones(n)
  for i in range(m):
    h = sigmoid(sum(dataMatrix[i]*weights))
    error = classLabels[i] - h
    weights = weights + alpha * error * dataMatrix[i]
  return weights

def randomGradAscentInprove(dataMatrix, classLabels, numIter=150):
  m, n = shape(dataMatrix)
  weights = ones(n)
  for j in range(numIter):
    dataIndex = range(m)
    for i in range(m):
      alpha = 4/(j+i+1.0)+0.01
      randIndex = int(random.uniform(0, len(dataIndex)))
      h = sigmoid(sum(dataMatrix[randIndex]*weights))
      error = classLabels[randIndex] - h
      weights = weights + alpha * error * dataMatrix[randIndex]
      del(dataIndex[randIndex])
  return weights


dataMat, labelMat = loadDataset()
print dataMat
print '\n', labelMat

# weights = gradAscent(dataMat, labelMat)
# weights = randomGradAscent(array(dataMat), labelMat)
weights = randomGradAscentInprove(array(dataMat), labelMat)
print '\n weights=\n',weights

# print plotBestFit(weights.getA())
print plotBestFit(weights)