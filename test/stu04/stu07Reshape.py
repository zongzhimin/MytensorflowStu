import tensorflow as tf

a = tf.random.uniform([4,28,28,3])
print(a.shape)
print(a.ndim)

b = tf.reshape(a,[4,784,3])
b = tf.reshape(a,[4,-1,3])

# 部分转置
a = tf.random.normal([4,3,2,1])
print(a.shape)

print(tf.transpose(a).shape)
# perm代指坐标的变化
print(tf.transpose(a,perm=[0,1,3,2]).shape)

# 增加维度
# 正在前增加，负在后增加
a = tf.random.normal([4,35,8])
print(tf.expand_dims(a,axis=0).shape)
print(tf.expand_dims(a,axis=3).shape)
print(tf.expand_dims(a,axis=-1).shape)
print(tf.expand_dims(a,axis=-4).shape)

# 去除维度为 1 的
print(tf.squeeze(tf.zeros([1,2,1,1,3])).shape)

a = tf.zeros([1,2,1,3])

tf.squeeze(a,axis=0)
tf.squeeze(a,axis=2)
tf.squeeze(a,axis=-2)
tf.squeeze(a,axis=-4)
