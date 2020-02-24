import tensorflow as tf
from tensorflow.keras import layers,Sequential

# drop out 在前向计算中w有几率本次计算以0代替（即不算本次w）

network = Sequential([
    layers.Dense(256,activation='relu'),
    layers.Dropout(0.5),  # 就像在上下层之间加了断路器
    layers.Dense(128,activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64,activation='relu'),
    layers.Dense(32,activation='relu'),
    layers.Dense(10)
])
x=tf.constant()
# 训练是指定training true
network(x,training=True)

# 测试时指定false 关闭
network(x,training=False)
