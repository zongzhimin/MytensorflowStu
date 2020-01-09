import tensorflow as tf

import tensorflow.keras as keras

net = keras.layers.Dense(10)
net.build((4, 8))
print(net.kernel)

print(net.bias)


x = tf.random.normal([4,784])
# 数据shape
print(x.shape)
net = keras.layers.Dense(10)
net.build((4,784))
# 最终shape
print(net(x).shape)
# 隐藏层shape
print(net.kernel.shape)
# bais
print(net.bias.shape)