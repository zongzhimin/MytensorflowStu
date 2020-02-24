import tensorflow as tf
from tensorflow.keras import layers

net = layers.BatchNormalization()
x = tf.random.normal([2, 3])
out = net(x)
print(net.trainable_variables)
print(net.variables)

x = tf.random.normal([2, 4, 4, 3], mean=1, stddev=0.5)
net = layers.BatchNormalization(axis=3)
out = net(x, training=True)

print(net.variables)

# 朝着x的mean 和 stddev 前进
for i in range(100):
    out = net(x, training=True)
print(net.variables)
