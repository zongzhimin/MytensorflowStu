import tensorflow as tf

a = tf.ones([2, 2])
# norm 二分数 所有元素平方和求平方根
print(tf.norm(a))

print(tf.sqrt(tf.reduce_sum(tf.square(a))))

# norm 可指定维度 如axis=1 ，则在列上做二分数（每一行进行计算）
print(tf.norm(a,ord=2,axis=1))

# norm 一分数
print(tf.norm(a,ord=1))
print(tf.norm(a,ord=1,axis=0))

a = tf.random.normal([4,10])
print(tf.reduce_min(a),tf.reduce_max(a),tf.reduce_mean(a))
print(tf.reduce_min(a,axis=1),tf.reduce_max(a,axis=1),tf.reduce_mean(a,axis=1))

# argmax 求某一维最大值的位置
print(a.shape)
# 默认维度axis=0
print(tf.argmax(a).shape)

a = tf.constant([1,2,3,4,5])
b = tf.range(5)
res = tf.equal(a,b)
print(res)
# 求两tensor相等元素个数
print(tf.reduce_sum(tf.cast(res,dtype=tf.int32)))
a = tf.reshape(tf.constant([0.1,0.2,0.7,0.9,0.05,0.05]),[2,3])
print(a)
pred = tf.cast(tf.argmax(a,axis=1),dtype=tf.int32)
print(pred)
y = tf.constant([2,1])

correct = tf.reduce_sum(tf.cast(tf.equal(pred,y),dtype=tf.int32))
print(correct)
print(correct/2)


# 去重 返回含下标
a = tf.constant([4,2,2,4,3])
b = tf.unique(a)
print(b)
print(b.y)
print(b.idx)

print(tf.gather(b.y,b.idx))
