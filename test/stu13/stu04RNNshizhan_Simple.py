import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers, datasets, Model
from tensorflow import keras
import os
import numpy as np

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 单层

# the most frequent words
total_words = 10000
max_review_len = 80
embedding_len = 100
batchsz = 64
(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=total_words)
# [b,80]
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_db = train_db.shuffle(10000).batch(batchsz, drop_remainder=True)  # 最后的batch长度不够则 drop

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.batch(batchsz, drop_remainder=True)

print('x_train:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test:', x_test.shape)


class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()

        # [b,64]
        self.state0 = [tf.zeros([batchsz, units])]
        self.state1 = [tf.zeros([batchsz, units])]
        # transform test to embedding representation
        # [b,80] => [b,80,100]
        self.embedding = layers.Embedding(total_words, output_dim=embedding_len, input_length=max_review_len)

        # [b,80,100] ,h_dim : 64
        # RNN : cell1 cell2 cell3
        # SimpleRNN
        self.rnn_cell0 = layers.SimpleRNNCell(units)
        self.rnn_cell1 = layers.SimpleRNNCell(units)

        # fc [b,80,100] => [b,64] => [b]
        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None):
        # [b,80]
        x = inputs
        # [b,80] => [b,80,100]
        x = self.embedding(x)
        # run cell compute
        # [b,80,100] => [b,64]
        state0 = self.state0
        for word in tf.unstack(x, axis=1):  # word:[b,100] 每次取b的单词的第i个一起
            # h1 = x*wxh + h0*whh
            out0, state0 = self.rnn_cell0(word, state0)
            out1, state1 = self.rnn_cell1(out0, state1)

        # out: [b,64] => [b,1]
        x = self.outlayer(out)
        # p(y is pos|x)
        prob = tf.sigmoid(x)

        return prob


def main():
    units = 64
    epochs = 4

    model = MyRNN(units)
    model.compile(optimizer=optimizers.Adam(lr=0.001), loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.fit(train_db, epochs=epochs, validation_data=test_db)


if __name__ == '__main__':
    main()
