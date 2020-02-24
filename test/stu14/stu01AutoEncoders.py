import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers,Model,Sequential,optimizers


tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

def save_images(imgs,name):
    new_im = Image.new('L',(28*32,28*32))

    index = 0
    for i in range(0,28*32,28):
        for j in range(0,28*32,28):
            im = imgs[index]
            im = Image.fromarray(im,mode='L')
            new_im.paste(im,(i,j))
            index +=1
    new_im.save(name)


h_dim = 20
batchsz = 512
lr = 1e-3

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train,x_test = x_train.astype(np.float32)/255.,x_test.astype(np.float32)/255.

train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batchsz * 5).batch(batchsz)

test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batchsz)

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


class AE(Model):
    def __init__(self):
        super(AE,self).__init__()

        # Encoders
        self.encoder = Sequential([
            layers.Dense(256,activation=tf.nn.relu),
            layers.Dense(128,activation=tf.nn.relu),
            layers.Dense(h_dim)
        ])

        # Decoders
        self.decoder = Sequential([
            layers.Dense(128,activation=tf.nn.relu),
            layers.Dense(256,activation=tf.nn.relu),
            layers.Dense(784)
        ])

    def call(self, inputs, training=None, mask=None):
        # [b,784] => [b,10]
        h = self.encoder(inputs)
        # [b,10] => [b,784]
        x_hat = self.decoder(h)
        return x_hat


model = AE()
model.build(input_shape=(None,784))
model.summary()

optimizer = optimizers.Adam(lr)

for epoch in range(100):
    for step,x in enumerate(train_db):

        x = tf.reshape(x,[-1,784])

        with tf.GradientTape() as tape:
            x_rec_logit = model(x)

            rec_loss = tf.losses.binary_crossentropy(x,x_rec_logit,from_logits=True)
            rec_loss = tf.reduce_mean(rec_loss)
        grads = tape.gradient(rec_loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))

        if step % 100 == 0:
            print(epoch,step,'loss=',float(rec_loss))

    # evaluation
    x = next(iter(test_db))
    x1 = tf.reshape(x, [-1, 784])
    logit = model(x1)
    x_hat = tf.sigmoid(logit)
    x_hat = tf.reshape(x_hat,[-1,28,28])

    x_concat = tf.concat([x,x_hat],axis=0)
    # print(x.shape)
    # print(x_concat.shape)
    # x_concat = x_hat
    x_concat = x_concat.numpy() * 255.
    x_concat = x_concat.astype(np.uint8)
    save_images(x_concat,'E:\\stuCode\\testData\\test1\\rec_epoch_%d.png'%epoch)

