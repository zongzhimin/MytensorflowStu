import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers, optimizers, datasets

(x,y),(x_test,y_test) = keras.datasets.mnist.load_data()
print('x:',x.shape)
print('y:',y.shape)
print(x.min()," ",x.max()," ",x.mean())
print('x test:',x_test.shape)
print('y test',y_test.shape)

print(y[:4])

y_onehot = tf.one_hot(y,depth=10)
print('y_onehot:',y_onehot.shape)
print(y_onehot[:2])
