import tensorflow as tf

from tensorflow.keras import layers,Sequential

x = tf.random.normal([4,80,100])
xt0 = x[:,0,:]

cell = layers.SimpleRNNCell(64)
cell2 = layers.SimpleRNNCell(64)
state0 = [tf.zeros([4,64])]
state1 = [tf.zeros([4,64])]

out0,state0 = cell(xt0,state0)
out1,state1 = cell2(out0,state1)

print(out1.shape)
print(state1[0].shape)

rnn = Sequential([
    layers.SimpleRNN(units=64,dropout=0.5,return_sequences=True,unroll=True),
    layers.SimpleRNN(units=64,dropout=0.5,unroll=True)
])
x = rnn(x)

print(x.shape)