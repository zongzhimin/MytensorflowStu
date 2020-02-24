import tensorflow as tf

a = tf.random.normal([3, 3])
print(a)
mask = a > 0
print(mask)
print(tf.boolean_mask(a, mask))
# where 返回True的下标
indices = tf.where(mask)
print(indices)
print(tf.gather(a, indices))

cond = tf.constant([[True, True, False], [True, False, False], [True, True, False]])
a = tf.ones([3, 3])
b = tf.zeros([3, 3])
# cond为true的位置从a取数，为false的位置从b取数
print(tf.where(cond, a, b))

indices = tf.constant([[4], [3], [1], [7]])
updates = tf.constant([9, 10, 11, 12])
shape = tf.constant([8])
print(shape)

# 对shape进行更新，indices为坐标列表，update为待更新值列表
scatter = tf.scatter_nd(indices, updates, shape)
print(scatter)

# 多维
indices = tf.constant([[0],[2]])
updates = tf.constant([[[5,5,5,5],[6,6,6,6],[7,7,7,7],[8,8,8,8]],[[5,5,5,5],[6,6,6,6],[7,7,7,7],[8,8,8,8]]])
print(updates.shape)
shape = tf.constant([4,4,4])
scatter = tf.scatter_nd(indices,updates,shape)
print(scatter)
