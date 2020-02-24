import tensorflow as tf

# MSE
# loss 函数：y 与 out 差值平方和
# L2-norm ： 开根号
y = tf.constant([1,2,3,0,2])
y = tf.one_hot(y,depth=4)
y = tf.cast(y,dtype=tf.float32)

out = tf.random.normal([5,4])

loss1 = tf.reduce_mean(tf.square(y-out))
loss2 = tf.square(tf.norm(y-out))/(5*4)
loss3 = tf.reduce_mean(tf.losses.MSE(y,out))
print(loss1)
print(loss2)
print(loss3)

print('------------------')
print('------------------')

# Entropy 熵 衡量数据的稳定性 与 信息量
# 熵越低，数据越不稳定，信息量越大
# 计算如下
a = tf.fill([4],0.25)
print(a)
print(-tf.reduce_sum(a*tf.math.log(a)/tf.math.log(2.)))
print('------------------')
a = tf.constant([0.1,0.1,0.1,0.7])
print(a)
print(-tf.reduce_sum(a*tf.math.log(a)/tf.math.log(2.)))
print('------------------')
a = tf.constant([0.01,0.01,0.1,0.97])
print(a)
print(-tf.reduce_sum(a*tf.math.log(a)/tf.math.log(2.)))

# 交叉熵 Cross Entroy 可作为loss函数
print(tf.losses.categorical_crossentropy([0,1,0,0],[0.25,0.25,0.25,0.25]))
print(tf.losses.categorical_crossentropy([0,1,0,0],[0.1,0.1,0.1,0.8]))
print(tf.losses.categorical_crossentropy([0,1,0,0],[0.1,0.1,0.7,0.1]))
print(tf.losses.categorical_crossentropy([0,1,0,0],[0.01,0.97,0.01,0.01]))

print(tf.losses.BinaryCrossentropy()([1],[0.1]))
print(tf.losses.binary_crossentropy([1],[0.1]))


# sigmoid + MSE 在两边预测错误时梯度小
# cross entroy 在错误时梯度很大 前期收敛快
# 具体使用 实践检验选取

# softmax + cross entroy 一起用中间肯定出现数值不稳定的情况
# tensforflow提供一个何在一起的函数并优化了数值不稳定的情况
x = tf.random.normal([1,784])
w = tf.random.normal([784,2])
b = tf.zeros([2])

logits = x@w +b
print(logits)

prob = tf.math.softmax(logits,axis=1)
print(prob)

# 建议这个
print(tf.losses.categorical_crossentropy([0,1],logits,from_logits=True))


# 不建议这个，数值不稳定
print(tf.losses.categorical_crossentropy([0,1],prob))
