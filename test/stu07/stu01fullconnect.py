import tensorflow as tf
from tensorflow import keras

# [4,784] => [4,512]
x = tf.random.normal([4,784])
net = tf.keras.layers.Dense(512)
out = net(x)
# [4,512]
print(out.shape)
# w [784,512]
print(net.kernel.shape)
# b (512,)
print(net.bias.shape)

# net需要初始化
net = tf.keras.layers.Dense(10)
# print(net.bias)
print(net.get_weights())
print(net.weights)

# 可通过build初始化
net.build(input_shape=(None,4))
# [4,10]
print(net.kernel.shape)
print(net.bias.shape)

net.build(input_shape=(None,20))
# [20,10]
print(net.kernel.shape)
print(net.bias.shape)

net.build(input_shape=(2,4))
# [4,10]
print(net.kernel.shape)
print(net.bias.shape)

# 多层
x = tf.random.normal([2,4])
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2,activation='relu'),
    tf.keras.layers.Dense(2,activation='relu'),
    tf.keras.layers.Dense(2)
])
model.build(input_shape=[None,4])
# 就是print打印参数
model.summary()

for p in model.trainable_variables:
    print(p.name,p.shape)
