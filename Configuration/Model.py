from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

#input
x = tf.placeholder(tf.float32, [None, 784])#784...
label = tf.placeholder(tf.float32, [None, 10])

#hidden weight and bias
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#hidden output
netoutput = tf.matmul(x,w) + b

#softmax output
y = tf.nn.softmax(netoutput)

#cost function for mini batch
crossentropy = -tf.reduce_sum(label*tf.log(y))

#train sgd
train = tf.train.GradientDescentOptimizer(0.01).minimize(crossentropy)
