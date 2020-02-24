import tensorflow as tf


# output [b,N]  b个预测数据
# target [b] 数组
def top_k_accurance(output, target, topk=(1,)):
    maxk = max(topk)
    # 总个数
    batch_size = target.shape[0]

    # 最前N大坐标即为预测值
    pred = tf.math.top_k(output, maxk).indices
    # 转置
    pred = tf.transpose(pred, perm=[1, 0])
    target_ = tf.broadcast_to(target, pred.shape)
    correct = tf.equal(pred, target_)

    res = []
    for k in topk:
        # a[:k] 取数列前k行，   tf.reshape(a,[-1])数组转数列
        correct_k = tf.cast(tf.reshape(correct[:k], [-1]), dtype=tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k * (100 / batch_size))
        res.append(acc)
    return res


output = tf.random.normal([10, 6])
output = tf.math.softmax(output, axis=1)
target = tf.random.uniform([10], maxval=6, dtype=tf.int32)
print('prob:', output.numpy())
pred = tf.argmax(output, axis=1)
print('pred:', pred.numpy())
print('label:', target.numpy())

acc = top_k_accurance(output, target, topk=(1, 2, 3, 4, 5, 6))
print('top k accurance', acc)