import tensorflow as tf

from tensorflow.keras import datasets,layers,Sequential,optimizers,metrics
#交叉验证集


def preprocess(x,y):
    x = tf.cast(x,dtype=tf.float32)/255.
    y = tf.cast(y,dtype=tf.int32)
    return x,y


batchsz = 128
# 11111
(x, y),(x_test, y_test) = datasets.mnist.load_data()
x_train,x_val = tf.split(x,num_or_size_splits=[50000,10000])
y_train,y_val = tf.split(y,num_or_size_splits=[50000,10000])

db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
db_train = db_train.map(preprocess).shuffle(10000).batch(batchsz)

db_val = tf.data.Dataset.from_tensor_slices((x_val,y_val))
db_val = db_val.map(preprocess).shuffle(10000).batch(batchsz)

db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.map(preprocess).batch(batchsz)

# 22222
# x_train 每次随机取一部分训练另一部分做val
(x, y),(x_test, y_test) = datasets.mnist.load_data()

for epoch in range(500):
    idx = tf.range(60000)
    idx = tf.random.shuffle(idx)
    x_train,y_train = tf.gather(x,idx[:50000]),tf.gather(y,idx[50000])
    x_val,y_val = tf.gather(x,idx[-10000:]),tf.gather(y,idx[-10000:])

    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    db_train = db_train.map(preprocess).shuffle(10000).batch(batchsz)

    db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    db_val = db_val.map(preprocess).shuffle(10000).batch(batchsz)

    # training...

    # evalutation...


# 框架中提供便捷的方法

batchsz = 128
(x, y), (x_test, y_test) = datasets.mnist.load_data()

db_train = tf.data.Dataset.from_tensor_slices((x,y))
db_train = db_train.map(preprocess).shuffle(60000).batch(batchsz)
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
ds_test = ds_test.map(preprocess).batch(batchsz)

network = Sequential([layers.Dense(256, activation='relu'),
                     layers.Dense(128, activation='relu'),
                     layers.Dense(64, activation='relu'),
                     layers.Dense(32, activation='relu'),
                     layers.Dense(10)])
network.build(input_shape=(None, 28*28))

network.compile(optimizer=optimizers.Adam(lr=0.01),
		loss=tf.losses.CategoricalCrossentropy(from_logits=True),
		metrics=['accuracy']
	)

# 传入数据时指定
# 每次会取90%训练 10%验证
network.fit(db_train,epochs=10,validation_split=0.1,validation_freq=2)
