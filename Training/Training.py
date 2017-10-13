from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf

sys.path.append('../')

import Configuration.Model as model
import Data.Datainput as dataset

#init op
init = tf.initialize_all_variables()

#session
sess = tf.Session()
sess.run(init)

#dataset
print('preparing dataset')
mnist = dataset.read_data_sets("MNIST_data/", one_hot=True)

#train
for i in range(1000):
    print ("ite = %d", i)
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(model.train, feed_dict={model.x: batch_x, model.label: batch_y})
    predict = tf.equal(tf.argmax(model.y,1), tf.argmax(model.label,1))
    accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))
    #print (sess.run(accuracy, feed_dict={x: mnist.test.images, label: mnist.test.labels}))
    with sess.as_default():
        print(accuracy.eval(feed_dict={model.x: mnist.test.images, model.label: mnist.test.labels}))

