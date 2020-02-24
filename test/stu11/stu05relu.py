import tensorflow as tf
from tensorflow.keras import layers
# 卷积神经网络relu层

x = tf.random.normal([2,3])

print(tf.nn.relu(x))

print(layers.ReLU()(x))
