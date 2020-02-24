import tensorflow as tf

a = tf.linspace(-6.,6.,10)
# sigmoid函数  1/(1+tf.math.exp(-x))
# 结果映射成0到1之间的数值
a = tf.sigmoid(a)
print(a)

x = tf.random.normal([1,28,28])*5
print(tf.reduce_min(x))
print(tf.reduce_max(x))
x = tf.sigmoid(x)
print(tf.reduce_min(x))
print(tf.reduce_max(x))

# sigmoid不能保证所有结果（0-1之间的数）数值和为1
# softmax可以
a = tf.linspace(-2.,2,5)
print(a)
print(tf.sigmoid(a))
print(tf.nn.softmax(a))

# tanh 可以映射到-1 到 1
# 是sigmod的y放大和平移
a = tf.constant([-2.,-1.,0.,1.,2.,])
print(tf.tanh(a))

