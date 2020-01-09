import tensorflow as tf

with tf.device("cpu"):
    a=tf.constant([1])
with tf.device("gpu"):
    b=tf.range(4)
print(a.device)
print(b.device)
