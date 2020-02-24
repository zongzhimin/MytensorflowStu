import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


# 数据准备
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
print(x.shape)
print(y.shape)
print(x_test.shape)
print(y_test.shape)
batchsz = 128

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(10000).batch(batchsz)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batchsz)

# db_iter = iter(db)
# sample = next(db_iter)
# print('batch:', sample[0].shape, sample[1].shape)

# 构建网络

model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),  # [b,784] => [b,256]
    layers.Dense(128, activation=tf.nn.relu),  # [b,256] => [b,128]
    layers.Dense(64, activation=tf.nn.relu),  # [b,128] => [b,64]
    layers.Dense(32, activation=tf.nn.relu),  # [b,64] => [b,32]
    layers.Dense(10)  # [b,32] => [b,10]
])
model.build(input_shape=[None, 28 * 28])  # 构建
model.summary()  # 调试，打印网络信息

# 优化器：完成w = w - lr*grad的过程
optimizer = optimizers.Adam(lr=1e-3)


def main():
    for epoch in range(30):
        for step, (x, y) in enumerate(db):
            # x: [b,28,28] => [b,784]
            # y: [b]
            x = tf.reshape(x, [-1, 28 * 28])
            with tf.GradientTape() as tape:
                # [b,784] => [b,10]
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=10)
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))
            grads = tape.gradient(loss_ce, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss', float(loss_ce), float(loss_mse))

        # test
        total_correct = 0
        total_sum = 0
        for x,y in db_test:
            # x: [b,28,28] => [b,784]
            # y: [b]
            x = tf.reshape(x, [-1, 28 * 28])
            # [b,10]
            logits = model(x)
            prob = tf.nn.softmax(logits)
            # pred: [b]
            pred = tf.argmax(prob,axis=1)
            pred = tf.cast(pred,tf.int32)
            correct = tf.reduce_sum(tf.cast(tf.equal(pred,y),tf.int32))
            total_correct += int(correct)
            total_sum += x.shape[0]
        acc = total_correct/total_sum
        print(epoch,'acc:',acc)


if __name__ == '__main__':
    main()
