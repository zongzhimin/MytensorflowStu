import tensorflow as tf

a = tf.linspace(-1.,1.,10)
print(a)
# 返回大于0的
# 大于0的梯度1 有利于反向传播
print(tf.nn.relu(a))
# 小于0 返回kx 大于0返回x
print(tf.nn.leaky_relu(a))