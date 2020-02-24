import tensorflow as tf
from tensorflow import keras

(x,y),(x_test,y_test) = keras.datasets.cifar10.load_data()

print('x:',x.shape)
print('y:',y.shape)
print(x.min()," ",x.max()," ",x.mean())
print('x test:',x_test.shape)
print('y test',y_test.shape)

# 方便迭代取值
db = tf.data.Dataset.from_tensor_slices(x_test)
print(next(iter(db)).shape)

db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
print(next(iter(db))[0].shape)

db = tf.data.Dataset.from_tensor_slices((x_test,y_test))

# daraset自带的打散功能
db = db.shuffle(10000)

# dataset预处理
def preprocess(x,y):
    x = tf.cast(x,dtype=tf.float32)/255.
    y = tf.cast(y,dtype=tf.int32)
    y = tf.one_hot(y,depth=10)
    return x,y


db2 = db.map(preprocess)
res = next(iter(db2))
print(res[0].shape)
print(res[1].shape)
print(res[1])
print(res[1][:2])

print('--------------------------')
# 一次取多个
db3 = db2.batch(32)
res = next(iter(db3))
print(res[0].shape)
print(res[1].shape)


# 多次从头开始
db_iter = iter(db3)
# catch StopTnteration 异常后给db_iter 重新赋值
# while True:
#    next(db_iter)

# 指定几次重复
db4 = db3.repeat(2)

