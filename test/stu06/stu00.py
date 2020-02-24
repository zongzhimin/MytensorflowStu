import tensorflow as tf

a = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
a = tf.reshape(a[:2],[-1])
print(a)
