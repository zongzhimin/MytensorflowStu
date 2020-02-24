import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 自动查看是否有mnist缓存数据集，如果没有会自动从google cloud 下载
# x: [60k, 28, 28]
# y: [60k]
(x, y), (x_test,y_test) = datasets.mnist.load_data()
# x归一化
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255
y = tf.convert_to_tensor(y, dtype=tf.int32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)
print(x.shape, y.shape, x.dtype, y.dtype)
print(tf.reduce_min(x), tf.reduce_max(x))
print(tf.reduce_min(y), tf.reduce_max(y))

# 迭代取值 每次128条
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)
print('batch:', sample[0].shape, sample[1].shape)

# [b, 784] => [b, 256] => [b, 128] => [b, 10]
# [dim_in, dim_out], [dim_out]
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3
for epoch in range(100):  # iterate db for 10
    for step, (x, y) in enumerate(train_db):  # for every batch
        # x:[128, 28, 28]
        # y:[128]
        # [b,28,28] => [b,28*28]
        x = tf.reshape(x, [-1, 28 * 28])
        with tf.GradientTape() as tape:
            # h1 = x@w1 + b1
            # [b,784]@[784,256] + [256] (此处broadcast_to可省略，会自动broadcast)
            h1 = x @ w1 + tf.broadcast_to(b1, [x.shape[0], 256])
            h1 = tf.nn.relu(h1)
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            out = h2 @ w3 + b3

            # compute loss
            # out: [b,10]
            # y:[b]
            y_onehot = tf.one_hot(y, depth=10)

            # mse = mean(sum((y-out)^2))
            # [b,10]
            loss = tf.square(y_onehot - out)
            # mean
            loss = tf.reduce_mean(loss)

        # 计算梯度
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # w1.assign_sub() 原地更新（保证w1依然是tf.Variable类型）
        # w1 = w1 - lr * grads[0]
        w1.assign_sub(lr * grads[0])
        # b1 = b1 - lr * grads[1]
        b1.assign_sub(lr * grads[1])
        # w2 = w2 - lr * grads[2]
        w2.assign_sub(lr * grads[2])
        # b2 = b2 - lr * grads[3]
        b2.assign_sub(lr * grads[3])
        # w3 = w3 - lr * grads[4]
        w3.assign_sub(lr * grads[4])
        # b3 = b3 - lr * grads[5]
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print(epoch, step, 'loss:', float(loss))

    # test/evluation
    # 使用当前 w1 b1 w2 b2 w3 b3
    total_correct,total_num = 0,0
    for step, (x, y) in enumerate(test_db):
        # [b,28,28] => [b,28*28]
        x = tf.reshape(x,[-1,28*28])
        # [b,784] => [b,256] => [b,128] => [b,10]
        h1 = tf.nn.relu(x@w1 + b1)
        h2 = tf.nn.relu(h1@w2 + b2)
        out = h2@w3 + b3

        # out [b,10]
        prob = tf.nn.softmax(out,axis=1)
        # [b,10] => [b]
        pred = tf.argmax(prob,axis=1)
        pred = tf.cast(pred,dtype=tf.int32)
        # y:[b]
        correct = tf.reduce_sum(tf.cast(tf.equal(pred,y),dtype=tf.int32))
        total_correct += int(correct)
        total_num += x.shape[0]
    acc = total_correct/total_num
    print('acc test: ',acc)
