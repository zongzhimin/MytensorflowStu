import tensorflow as tf
from tensorflow.keras import Sequential,optimizers

model = Sequential([
    # ...
])
optimizer = optimizers.Adam(lr=0.0001)
# 例如。。。。
x = tf.constant()
y = tf.constant()
with tf.GradientTape() as tape:
    logits = model(x)
    loss = tf.losses.categorical_crossentropy(y,logits,from_logits=True)

grads = tape.gradient(loss,model.trainable_variables)

# 梯度爆炸解决方法
# 梯度过大的，如大于15 则令（其/自己的模）*15
grads = [tf.clip_by_norm(g,15) for g in grads]
optimizer.apply_gradients(zip(grads,model.trainable_variables))
