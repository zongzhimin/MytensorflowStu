import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from PIL import Image
import glob
from stu15.stu04w_gan import Generator, Discriminator

from stu15.dataset import make_anime_dataset


def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    # toimage(final_image).save(image_path)
    img = Image.fromarray(np.uint8(final_image))
    img.save(image_path)


def celoss_ones(logits):
    # [b,1]  全是真的
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    # [b,1]  全是假的
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)

def gradient_penalty(discriminator,batch_x,fake_image):
    batchsz = batch_x.shape[0]
    # [b,w,h,c]
    t = tf.random.uniform([batchsz,1,1,1])
    t = tf.broadcast_to(t,batch_x.shape)

    interplate = t * batch_x + (1-t) * fake_image
    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplate_logit = discriminator(interplate)
    grads = tape.gradient(d_interplate_logit,interplate)
    # grads : [b,w,h,c] => [b,-1]
    grads = tf.reshape(grads,[grads.shape[0],-1])
    gp = tf.norm(grads,axis=1)  # [b]
    gp = tf.reduce_mean((gp-1)**2)
    return gp


def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    # treat real image as real
    # treat generated image as fake
    fake_image = generator(batch_z, training=is_training)
    d_fake_logits = discriminator(fake_image, training=is_training)
    d_real_logits = discriminator(batch_x, is_training)

    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)
    gp = gradient_penalty(discriminator,batch_x,fake_image)

    loss = d_loss_real + d_loss_fake + 1.*gp
    return loss,gp


def g_loss_fn(generator, discriminator, batch_z, is_training):
    fake_img = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_img, is_training)
    # 希望它能骗过 discriminator
    loss = celoss_ones(d_fake_logits)
    return loss


def main():
    tf.random.set_seed(22)
    np.random.seed(22)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')

    # hyper parameters
    z_dim = 100
    epochs = 3000000
    batch_size = 512
    learning_rate = 0.002
    is_training = True

    img_path = glob.glob(r'E:\stuCode\testData\faces\*.jpg')
    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size)
    print(dataset, img_shape)
    sample = next(iter(dataset))
    print(sample.shape)
    dataset = dataset.repeat()
    db_iter = iter(dataset)

    generator = Generator()
    generator.build(input_shape=(None, z_dim))
    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 64, 64, 3))

    g_optimezer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimezer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    # discriminator.load_weights(r'E:\stuCode\testData\w_gan_weights\d_weights.ckpt')
    # generator.load_weights(r'E:\stuCode\testData\w_gan_weights\g_weights.ckpt')

    for epoch in range(epochs):
        batch_z = tf.random.uniform([batch_size, z_dim], minval=-1., maxval=1.)
        batch_x = next(db_iter)
        # train discriminator
        with tf.GradientTape() as tape:
            d_loss,gp = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimezer.apply_gradients(zip(grads, discriminator.trainable_variables))

        # train generator
        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimezer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 100 == 0:
            print(epoch, 'd_loss:', float(d_loss), 'g_loss', float(g_loss),'gp:',float(gp))

            z = tf.random.uniform([100, z_dim], minval=-1., maxval=1.)
            fake_image = generator(z, training=False)
            img_path = r'E:\stuCode\testData\w_gan_test1\wgan-%d.png' % epoch
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')
            # discriminator.save_weights(r'E:\stuCode\testData\w_gan_weights\d_weights.ckpt')
            # generator.save_weights(r'E:\stuCode\testData\w_gan_weights\g_weights.ckpt')


if __name__ == '__main__':
    main()