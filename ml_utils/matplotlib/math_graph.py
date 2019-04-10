# 梯度上升函数
def gradAscent(dataMatIn, classLabels):
dataMat = mat(dataMatIn)
# 转化成矩阵 并转置
labelMat = mat(classLabels).transpose()
# print labelMat
# 获取行和列数
m, n = shape(dataMat)
# 设置步长
alpha = 0.001
# 设置最大迭代次数
maxCircle = 500
# 初始化权重系数
weights = ones((n, 1))
# 循环迭代
for k in range(maxCircle):
h = sigmoid(dataMat * weights)
# print h
error = (labelMat - h)
# print error
weights = weights + alpha * dataMat.transpose() * error
return weights








import numpy as np
# print np.abs(-1)
# import Tkinter
import matplotlib.pyplot as plt

x = np.arange(-8, 8, 0.1)
f = 1 / (1 + np.exp(-x))
plt.plot(x, f)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
