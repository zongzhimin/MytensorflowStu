import tensorflow as tf

# tf.broadcast_to 隐式复制，没有在内存中复制
# tf.tile 内存复制
a = tf.constant([[0,1,2],[3,4,5],[6,7,8]])
# [2,2] 第一维度复制为原来2倍，第二维度复制为原来二倍，低维开始
print(tf.tile(a,[2,2]))

# expand_dims + tile 可完成broadcast_to的操作，推荐用broadcast_to
# broadcast_to 内存占用少，且默写操作符会隐式broadcast不用显式调用
