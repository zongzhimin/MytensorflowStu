import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('GPU',tf.test.is_gpu_available())

a = tf.constant(2.)
b = tf.constant(4.)

print(a*b)

c = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
print(c[:2])
print(tf.argmax(c,axis=1))