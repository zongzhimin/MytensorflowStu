import tensorflow as tf

w = tf.constant(1.)
x = tf.constant(2.)
y = x * w

# 只可使用一次 tf.GradientTape(persistent=True) as type 则可以多次调用
# 要记得资源释放
with tf.GradientTape() as tape:
    tape.watch([w])
    y2 = x * w

grad1 = tape.gradient(y2,[w])
print(grad1)

w = tf.Variable(1.0)
b = tf.Variable(2.0)
x = tf.Variable(3.0)

# 二阶求导
with tf.GradientTape() as t1:
    with tf.GradientTape() as t2:
        y = w * x + b
    dy_dw,dy_db = t2.gradient(y,[w,b])
d2y_d2w = t1.gradient(dy_dw,w)

print(dy_dw)
print(dy_db)
print(d2y_d2w)

assert dy_dw.numpy() == 3.0
assert d2y_d2w is None
