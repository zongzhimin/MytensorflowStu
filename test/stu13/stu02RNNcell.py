import tensorflow as tf
from tensorflow.keras import layers

cell = layers.SimpleRNNCell(3)
cell.build(input_shape=(None,4))

print(cell.trainable_variables)

x = tf.random.normal([4,80,100])
xt0 = x[:,0,:]

cell = layers.SimpleRNNCell(64)
h0 = tf.zeros([4,64])

out,ht1 = cell(xt0,[h0])

print(out.shape)
print(ht1[0].shape)

# 是同一个
print(id(out))
print(id(ht1[0]))