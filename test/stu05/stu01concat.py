import tensorflow as tf

a = tf.ones([4,35,8])
b = tf.ones([2,35,8])

c = tf.concat([a,b],axis=0)
print(c.shape)

a = tf.ones([4,32,8])
b = tf.ones([4,3,8])
c = tf.concat([a,b],axis=1)
print(c.shape)


# stack创建新维度
a = tf.random.uniform([4,35,8])
b = tf.random.uniform([4,35,8])
c = tf.stack([a,b],axis=0)
print(c.shape)

# unstack 分割
aa,bb = tf.unstack(c,axis=0)
print(aa.shape)
print(bb.shape)
# 会全部打散
res = tf.unstack(c,axis=2)

# split分割，指定分割组数 或每组的一个数组
res = tf.split(c,axis=3,num_or_size_splits=2)

res = tf.split(c,axis=3,num_or_size_splits=[2,3,3])


