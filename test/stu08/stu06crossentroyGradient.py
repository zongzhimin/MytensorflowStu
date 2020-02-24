import tensorflow as tf

x = tf.random.normal([2,4])
w = tf.random.normal([4,3])
b = tf.zeros([3])
y = tf.constant([2,0])

with tf.GradientTape() as tape:
    tape.watch([w,b])
    logit = x@w + b
    loss = tf.losses.categorical_crossentropy(tf.one_hot(y,depth=3),logit,from_logits=True)
grads = tape.gradient(loss,[w,b])
print(grads[0])
print(grads[1])
