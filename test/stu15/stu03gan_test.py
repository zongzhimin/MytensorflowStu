import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from PIL import Image
import glob
from stu15.stu01gan import Generator, Discriminator

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


generator = Generator()
generator.load_weights(r'E:\stuCode\testData\gan_weights\g_weights.ckpt')
z = tf.random.uniform([100, 100], minval=-1., maxval=1.)
fake_image = generator(z,training=False)
img_path = r'E:\stuCode\testData\test5\gan_test.png'
save_result(fake_image.numpy(), 10, img_path, color_mode='P')
