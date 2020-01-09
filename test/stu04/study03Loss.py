import tensorflow as tf

out = tf.random.uniform([4,10])

y = tf.range(4)
print(y)
# 种类对应成向量
y = tf.one_hot(y,depth=10)
print(y)

loss = tf.keras.losses.mse(y,out)
print(loss)
loss = tf.reduce_mean(loss)
print(loss)
