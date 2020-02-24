import tensorflow as tf

a = tf.constant([0,1,2,3,4,5,6,7,8,9])
print(a)
# 值大于2的不变，小于2的全变成2
print(tf.maximum(a,2))
# 值小于8的不变，大于8的全变成8
print(tf.minimum(a,8))
# 前两者结合
print(tf.clip_by_value(a,2,8))

a = a - 5
# 大于0的不变小于0的变成0
print(tf.nn.relu(a))
print(tf.maximum(a,0))

a = tf.random.normal([2,2],mean=10)
print(a)
# a的模
print(tf.norm(a))
# 按比例修改，最终模为15
aa = tf.clip_by_norm(a,15)
print(aa)
print(tf.norm(aa))
