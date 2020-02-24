import tensorflow as tf
from tensorflow.keras import layers

x = tf.range(5)
x = tf.random.shuffle(x)
print(x)

# 总共待处理单词：10
# 每个词用4维向量表示
net = layers.Embedding(10,4)
out = net(x)
print(out.shape)

print(out)
print(net.trainable_variables)
