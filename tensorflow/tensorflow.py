#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import sys
sys.executable

import os
try:
	os.chdir(os.path.join(os.getcwd(), 'tensorflow'))
	print(os.getcwd())
except:
	pass


#%%
import tensorflow as tf
import numpy as np

#%%
trX = np.linspace(-1, 1, 101)
trY = 2 * trX + np.random.randn(*trX.shape)*0.33


#%%
X = tf.placeholder("float")
Y = tf.placeholder("float")
def model(X, w):
    return tf.multiply(X, w)


#%%
w = tf.Variable(0.0, name="weight")
y_model = model(X, w)


#%%
cost = tf.square(Y - y_model)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)


#%%
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(100):
        for (x, y) in zip(trX, trY):
            sess.run(train_op, feed_dict={X: x, Y: y})
    print sess.run(w)


#%%
from tensorflow.examples.tutorials.mnist import input_data
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


#%%
def model(X, w):
    return tf.matmul(X, x)


#%%
print (tf.__version__)



