import tensorflow as tf
from tensorflow.keras import layers

x = tf.random.normal([100,8,8,3])
# N=4 :生成4个
# 5*5 每次卷积块大小
# 滑步：2
# padding='same'自动补， padding='valid'不补
# C 在传入计算时生成
layer = layers.Conv2D(4,kernel_size=5,strides=2,padding='same')
out = layer(x)
print(out.shape)
