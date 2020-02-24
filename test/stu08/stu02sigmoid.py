import tensorflow as tf

a = tf.linspace(-10.,10.,10)
with tf.GradientTape() as tape:
    tape.watch(a)
    y = tf.sigmoid(a)
grads = tape.gradient(y,[a])
print(a)
print(y)
print(grads)

