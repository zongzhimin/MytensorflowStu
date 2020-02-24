import tensorflow as tf

a = tf.ones([1, 5, 5, 3])
print(a[0][0])
print(a[0][0][0])
print(a[0][0][0][0])

print(a[0, 0])
print(a[0, 0, 0])
print(a[0, 0, 0, 0])

a = tf.range(10)
print(a)
print(a[:-1])
print(a[-2:])
print(a[:2])
print(a[:-1])
print(a[0:8:2])
# step-1 实现倒叙一样的功能
print(a[::-1])

a = tf.random.uniform([4, 28, 28, 3])
print(a[0, ...])
print(a[..., 2])
print(a[0, ..., 2])
