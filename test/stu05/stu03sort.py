import tensorflow as tf

a = tf.random.shuffle(tf.range(5))
print(a)
# 排序
print(tf.sort(a,direction="DESCENDING"))
# 排序，返回的是索引
print(tf.argsort(a,direction="DESCENDING"))


# 维度为2是 是对每一行
a = tf.random.uniform([3,3],maxval=10,dtype=tf.int32)
print(a)
print(tf.sort(a))
print(tf.sort(a,direction="DESCENDING"))
print(tf.argsort(a))

# 前n个
res = tf.math.top_k(a,2)
print(res)
# 索引
print(res.indices)
# 值
print(res.values)


# top accurance
prob = tf.constant([[0.1,0.2,0.7],[0.2,0.7,0.1]])
target = tf.constant([2,0])

k_b = tf.math.top_k(prob,3).indices
print(k_b)
# 转置
k_b = tf.transpose(k_b,[1,0])

target = tf.broadcast_to(target,[3,2])
print(target)
