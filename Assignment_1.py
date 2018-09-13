#!/bin/python3.6
# Samuel Cheng
# CGML Assignment 1

import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


from tqdm import tqdm

NUM_FEATURES = 4
BATCH_SIZE = 32
NUM_BATCHES = 300


class Data(object):
    def __init__(self):
        num_samp = 50
        sigma = 0.1
        np.random.seed(31415)

        self.index = np.arange(num_samp)
        self.x = np.random.uniform(size=(num_samp, 1))
        self.y = np.sin(2*np.pi*self.x) + sigma * np.random.normal(size=(num_samp, 1))


"""
    def get_batch(self):
        choices = np.random.choice(self.index, size=BATCH_SIZE)

        return self.x[choices], self.y[choices].flatten()
"""

def f(x):
    w = tf.get_variable('w', [NUM_FEATURES, 1], tf.float32,
                        tf.random_normal_initializer())
    b = tf.get_variable('b', [], tf.float32, tf.zeros_initializer())
    mu = tf.get_variable('mu', [NUM_FEATURES, 1], tf.float32, tf.random_normal_initializer())
    sig = tf.get_variable('sig', [NUM_FEATURES, 1], tf.float32, tf.random_normal_initializer())
    return tf.transpose(tf.matmul(tf.transpose(w), tf.exp((-1)*tf.pow((tf.transpose(x)-mu), 2)/tf.pow(sig, 2))) + b)


x = tf.placeholder(tf.float32,[50,1])
y = tf.placeholder(tf.float32,[50,1])
y_hat = f(x)

loss = tf.reduce_mean(tf.pow(y_hat - y, 2))
optim = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

data = Data()
for i in range(0,3000):
    loss_np, _ = sess.run([loss, optim], feed_dict={x: data.x, y: data.y})

weightholder = []

print("Parameter estimates:")
for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    weightholder.append(np.array(sess.run(var)).flatten())
    print(
        var.name.rstrip(":0"),
        np.array_str(np.array(sess.run(var)).flatten(), precision=3))

w1 = weightholder[0]
b1 = weightholder[1]
mu1 = weightholder[2]
sig1 = weightholder[3]

x1 = np.linspace(0,1,100,dtype=np.float32)
y1 = np.sin(2*np.pi*x1)

y_hat1 = (w1[0] * tf.exp((-1)*tf.pow((x1-mu1[0]), 2)/tf.pow(sig1[0], 2)) + w1[1] * tf.exp((-1)*tf.pow((x1-mu1[1]), 2)/tf.pow(sig1[1], 2))
          + w1[2] * tf.exp((-1)*tf.pow((x1-mu1[2]), 2)/tf.pow(sig1[2], 2)) + w1[3] * tf.exp((-1)*tf.pow((x1-mu1[3]), 2)/tf.pow(sig1[3], 2))
          + b1
          )


x2 = np.linspace(0, 1, 100)
print(mu1)
plt.figure(1)

plt.scatter(data.x, data.y)
plt.plot(x1, y1)
plt.plot(x1, sess.run(y_hat1), 'r--')
plt.ylabel('y')
plt.xlabel('x')
plt.title('Base Function, Training Data and Trained Model')

plt.figure(2)
plt.plot(x2, mlab.normpdf(x2, mu1[0], sig1[0]), label='gaussian 1')
plt.plot(x2, mlab.normpdf(x2, mu1[1], sig1[1]), label='gaussian 2')
plt.plot(x2, mlab.normpdf(x2, mu1[2], sig1[2]), label='gaussian 3')
plt.plot(x2, mlab.normpdf(x2, mu1[3], sig1[3]), label='gaussian 4')
plt.ylabel('y')
plt.xlabel('x')
plt.title('Gaussian Bases for Fit')
plt.show()
