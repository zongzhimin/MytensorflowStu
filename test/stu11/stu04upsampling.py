import tensorflow as tf
from tensorflow.keras import layers
x = tf.random.normal([1,7,7,4])

layer = layers.UpSampling2D(size=3)
out = layer(x)
print(out.shape)

layer = layers.UpSampling2D(size=2)
out = layer(x)
print(out.shape)

