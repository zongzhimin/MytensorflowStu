import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential, optimizers

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


def save_images(imgs, name):
    new_im = Image.new('L', (28 * 10, 28 * 10))

    index = 0
    for i in range(0, 28 * 10, 28):
        for j in range(0, 28 * 10, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1
    new_im.save(name)


h_dim = 20
batchsz = 512
lr = 1e-3

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.

train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batchsz * 5).batch(batchsz)

test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batchsz)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

z_dim = 10


class VAE(Model):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = layers.Dense(128, activation=tf.nn.relu)
        self.fc2 = layers.Dense(z_dim)  # mean
        self.fc3 = layers.Dense(z_dim)  # 方差

        # Decoder
        self.fc4 = layers.Dense(128, activation=tf.nn.relu)
        self.fc5 = layers.Dense(784)

    def encoder(self, x):
        h = self.fc1(x)
        mu = self.fc2(h)
        log_var = self.fc3(h)
        return mu, log_var

    def decoder(self, z):
        out = self.fc4(z)
        out = self.fc5(out)
        return out

    def reparameterize(self, mu, log_var):
        eps = tf.random.normal(log_var.shape)
        std = tf.exp(log_var*0.5)
        z = mu + std * eps
        return z

    def call(self, inputs, training=None, mask=None):
        # [b,784] => [b, z_dim], [b, z_dim]
        mu, log_var = self.encoder(inputs)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var


model = VAE()
# model.build(input_shape=(None,784))
optimizer = optimizers.Adam(lr=lr)

for epoch in range(1000):
    for step, x in enumerate(train_db):
        x = tf.reshape(x, [-1, 784])
        with tf.GradientTape() as tape:
            x_rec_logits, mu, log_var = model(x)
            rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec_logits)
            rec_loss = tf.reduce_sum(rec_loss) / x.shape[0]

            # compute kl divergence (mu,var) ~ (0,1)
            # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
            kl_div = -0.5 * (log_var + 1 - mu ** 2 - tf.exp(log_var))
            kl_div = tf.reduce_sum(kl_div) / x.shape[0]
            loss = rec_loss + 1. * kl_div
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, 'kl_div:', float(kl_div), 'rec_loss', float(rec_loss))

    # evaluation
    z = tf.random.normal((batchsz, z_dim))
    logits = model.decoder(z)
    x_hat = tf.sigmoid(logits)
    x_hat = tf.reshape(x_hat, shape=[-1, 28, 28]).numpy() * 255.
    x_hat = x_hat.astype(np.uint8)
    save_images(x_hat, 'E:\\stuCode\\testData\\test2\\rec_epoch_%d.png' % epoch)

    # x = next(iter(test_db))
    # x_hat_logits, mu, log_var = model(tf.reshape(x, [-1, 784]))
    # x_hat = tf.sigmoid(x_hat_logits)
    # x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
    # x_hat = x_hat.astype(np.uint8)
    # save_images(x_hat, 'E:\\stuCode\\testData\\test3\\rec_epoch_%d.png' % epoch)

    x = next(iter(test_db))
    x = tf.reshape(x, [-1, 784])
    x_hat_logits, _, _ = model(x)
    x_hat = tf.sigmoid(x_hat_logits)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
    x_hat = x_hat.astype(np.uint8)
    save_images(x_hat, 'E:\\stuCode\\testData\\test3\\rec_epoch_%d.png' % epoch)
