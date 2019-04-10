#%%
msg = "HelloWord"
print msg

import numpy as np
import tensorflow as tf

a = tf.constant(1)
b = tf.constant(2)

c = a * b

sess = tf.Session()

print(sess.run(c))

# a.graph可以查询张量的计算图，未指定的情况下就是默认计算图
print (a.graph is tf.get_default_graph())

# 定义了两个计算图：一个将变量v初始化为0，一个将变量v初始化为1；运行不同的计算图，得到不同的变量值
# 体现了计算图可以用来隔离张量和计算；此外还可以提供管理张量和计算的机制。
g1 = tf.Graph()
with g1.as_default():
  v = tf.get_variable("v", shape=[1], initializer=tf.zeros_initializer())

g2 = tf.Graph()
with g2.as_default():
  v = tf.get_variable("v", shape=[1], initializer=tf.ones_initializer())

with tf.Session(graph=g1) as sess:
  tf.initialize_all_variables().run()
  with tf.variable_scope("", reuse=True):
    print (sess.run(tf.get_variable("v")))

with tf.Session(graph=g2) as sess:
  tf.initialize_all_variables().run()
  with tf.variable_scope("", reuse=True):
    print (sess.run(tf.get_variable("v")))

# 张量中保存的是计算过程，只是对运算结果的引用；tensorflow计算的结果不是一个具体的数字，而是一个
# 张量的结构。一个张量主要包含三种属性：名字、结构、类型。通过会话（Session）来执行定义好的运算
# 全连接网络的前向传播算法：一个最简单的神经元结构的输出就是所有输入发的加权和，而不同输入的权重
# 就是神经元的参数。神经网络的优化过程就是优化神经元中的参数获取。【全连接是表示相邻两层之间任意两个
# 节点之间都有连接】。

# 随机数生成函数：正态分布的随机数


# w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
# w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# x = tf.constant([[0.7, 0.9]])

# a = tf.matmul(x, w1)
# y = tf.matmul(a, w2)

# sess = tf.Session()
# sess.run(w1.initializer)  
# sess.run(w2.initializer)  
# print(sess.run(y))  
# sess.close()

# with tf.Session() as sess:
#   sess.run(w1.initializer)
#   sess.run(w2.initializer)
#   print (sess.run(y))




#%%
# 均方根误差（均方的基础上开了根号，意义一样）
# 均方误差（MSE）度量的是预测值和真实值之间差的平方的均值，它只考虑误差的平均大小
# 不考虑其方向。但经过平方之后，偏离真实值较多的预测值会受到更严重的惩罚，并且MSE的数学
# 特性很好，也易于求导，对于计算梯度也比较方便。
def rmse(predictions, targets):
  differences = predictions - targets
  differences_squared = differences ** 2
  mean_of_differences_squared = differences_squared.mean()
  rmse_val = np.sqrt(mean_of_differences_squared)
  return rmse_val

y_hat = np.array([0.000, 0.166, 0.333])
y_true = np.array([0.000, 0.254, 0.998])

print ("d is: " + str(["%.8f" % elem for elem in y_hat]))
print ("p is: " + str(["%.8f" % elem for elem in y_true]))

rmse_val = rmse(y_hat, y_true)
print ("rms error is: " + str(rmse_val))

#%%
# 平方绝对误差/L1误差
# 平均绝对误差（MAE）度量的是预测值和实际观测值之间绝对差之和的平均值。和MSE一样，
# 这种度量方法是在不考虑方向的情况下的，MAE需要像线性规划这样更复杂的工具来计算梯度。
# 此外，MAE对于异常值更加稳重，因为它不使用平方。

def mae(predictions, targets):
  differences = predictions - targets
  absolute_differences = np.abs(differences)
  mean_absolute_differences = absolute_differences.mean()
  return mean_absolute_differences

y_hat = np.array([0.000, 0.166, 0.333])
y_true = np.array([0.000, 0.254, 0.998])

print ("d is: " + str(["%.8f" % elem for elem in y_hat]))
print ("p is: " + str(["%.8f" % elem for elem in y_true]))

rmae_val = mae(y_hat, y_true)
print ("rms error is: " + str(rmae_val))

#%%
# 平均偏差误差（mean bias error）运用的较少  跟MAE很像，只是公式中没有绝对值


