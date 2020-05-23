import tensorflow as tf
from tensorflow.keras import layers

x = tf.random.normal([1,14,14,4])
# layers 自带
pool = layers.MaxPool2D(2,strides=2)
out = pool(x)
print(out.shape)

pool = layers.MaxPool2D(MaxPool2D=(2,2),strides=(1,1),padding='same')
out = pool(x)
print(out.shape)

pool = layers.MaxPool2D(3,strides=2)
out = pool(x)
print(out.shape)

# 函数
out = tf.nn.max_pool2d(x,2,strides=2,padding='VALID')
print(out.shape)
