import tensorflow as tf
import numpy as np

a = tf.convert_to_tensor(np.ones([2, 3]))
print(a)

b = tf.convert_to_tensor([1, 2])
print(b)

c = tf.convert_to_tensor([[1.], [2, ]])
print(c)

print(tf.zeros([]))

print(tf.zeros([1]))

print(tf.zeros([2]))

print(tf.zeros([2, 2]))

print(tf.zeros([2, 3, 3]))

d = tf.zeros([2, 3, 3])

e = tf.zeros_like(d)
print(e)

print(tf.zeros(d.shape))

tf.ones(1)
tf.ones([])
tf.ones([1])

tf.fill([2, 3], 0)
tf.fill([1, 2], 3)

print(tf.fill([3, 4], 8))

# 随机初始化

# 正态分布(N(1,1))
print(tf.random.normal([2,2],mean=1,stddev=1))
# 截断的正态分布（在中间）
tf.random.truncated_normal([2,2],mean=0,stddev=1)

# 均匀分布
print(tf.random.uniform([2,2],minval=0,maxval=1))


# 数据打散
# 连续数（对应下标）
idx = tf.range(10)
# 打散坐标
idx = tf.random.shuffle(idx)
print(idx)
# 数据
a = tf.random.normal([10,784])
b = tf.random.uniform([10],maxval=10,dtype=tf.int32)
# print(a)
print(b)
# 打散数据（根据坐标）
a = tf.gather(a,idx)
b = tf.gather(b,idx)
# print(a)
print(b)

print(tf.constant(1))

print(tf.constant([1]))

print(tf.constant([1,2.]))

