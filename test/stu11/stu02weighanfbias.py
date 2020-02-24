import tensorflow as tf
from tensorflow.keras import layers

# 框架带的
x = tf.random.normal([100,32,32,3])
layer = layers.Conv2D(4,kernel_size=5,strides=2,padding='same')
out = layer(x)
print(out.shape)
print(layer.kernel.shape)
print(layer.bias.shape)


print('-----------------')


# 自己运算
x = tf.random.normal([1,32,32,3])
w = tf.random.normal([5,5,3,4])
b = tf.zeros([4])

out = tf.nn.conv2d(x,w,strides=1,padding='VALID')
out = out +b
print(out.shape)

out = tf.nn.conv2d(x,w,strides=2,padding='VALID')
print(out.shape)


