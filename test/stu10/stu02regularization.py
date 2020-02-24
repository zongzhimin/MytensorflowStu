import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, Sequential, models

numwords = 1000
# 添加正则化约束
keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularization.l2(0.01), activation=tf.nn.relu,
                       input_shape=(numwords,))
])

# 2在for循环的 with 梯度计算里面
network = Sequential([]) # network初始化
loss = 0  # 。。。
loss_regularization = []
for p in network.trainable_variables:
    loss_regularization.append(tf.nn.l2_loss(p))
loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))
loss = loss + 0.001*loss_regularization

