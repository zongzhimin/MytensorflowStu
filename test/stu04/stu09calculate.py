import tensorflow as tf

a = tf.ones([2,2])
b = tf.fill([2,2],2.)

# 对应位置加减乘除
print(a + b)
print(a - b)
print(a * b)
print(a / b)

# 整除，取余
print(b // a)
print(b % a)

# 对数
# tf.math.log 是loge（Ln） 没有log10  log2
tf.math.log(a)
# 计算log10  log2 可以如下
# log 10
print(tf.math.log(a)/tf.math.log(10.))
# log 2
print(tf.math.log(b)/tf.math.log(2.))
# 指数 不在math下
print(tf.exp(a))

print(tf.pow(b,3))
print(b**3)

print(tf.sqrt(b))

# 矩阵相乘
print(a @ b)
print(tf.matmul(a,b))

a = tf.ones([4,2,3])
b = tf.fill([4,3,5],2.)

# 多维度并行计算
print((a@b).shape)
print(tf.matmul(a,b).shape)

c = tf.fill([3,5],3.)
cc = tf.broadcast_to(b,[4,3,5])
print(a@cc)

x = tf.ones([4,2])
w = tf.ones([2,1])
b = tf.constant(0.1)

out = x@w + b
print(out)

out = tf.nn.relu(out)
print(out)
