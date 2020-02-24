import tensorflow as tf

a = tf.reshape(tf.range(9),[3,3])
print(a)

# 行（上下）、列（左右）填充的个数
print(tf.pad(a,[[0,0],[0,0]]))
print(tf.pad(a,[[1,0],[0,0]]))
print(tf.pad(a,[[1,1],[0,0]]))
print(tf.pad(a,[[1,1],[1,0]]))
print(tf.pad(a,[[1,1],[1,1]]))

b = tf.random.normal([4,28,28,3])
b = tf.pad(b,[[0,0],[2,2],[2,2],[0,0]])
print(b.shape)
