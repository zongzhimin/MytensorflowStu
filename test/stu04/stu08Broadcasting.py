import tensorflow as tf

# broadcasing 右端（低维）对齐 没有复制内存

x = tf.random.uniform([4, 32, 32, 3])
# broadcasting 隐式
print((x + tf.random.normal([3])).shape)

# 显示
b = tf.broadcast_to(tf.random.normal([4,1,1,1]),[4,32,32,3])

print(b.shape)

# tile 复制内存
a = tf.ones([3,4])
a2 = tf.expand_dims(a,axis=0)
print(a2.shape)

# 2,1,1 指代各维度复制的倍数
a2 = tf.tile(a2,[2,1,1])
print(a2.shape)
print("----------------")
print(tf.version.VERSION)

