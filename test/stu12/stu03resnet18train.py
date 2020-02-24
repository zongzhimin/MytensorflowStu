import tensorflow as tf
from tensorflow.keras import layers, datasets, optimizers, Sequential
import os
from stu12.stu02resnetshizhan import resnet18
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y,axis=1)
y_test = tf.squeeze(y_test,axis=1)

print(x.shape,y.shape,x_test.shape,y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.map(preprocess).shuffle(10000).batch(64)

test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(64)

sample = next(iter(train_db))
print('sample :',sample[0].shape,sample[1].shape,tf.reduce_mean(sample[0]),tf.reduce_max(sample[0]))


def main():

    # [b,32,32,3] => [b,1,1,512]
    model = resnet18()
    model.build(input_shape=(None, 32,32,3))
    optimizer = optimizers.Adam(lr=1e-4)

    for epoch in range(50):
        for step,(x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                # [b,32,32,3] = > [b,100]
                logits = model(x)
                # [b] => [b,100]
                y_onehot = tf.one_hot(y,depth=100)
                # compute loss
                loss = tf.losses.categorical_crossentropy(y_onehot,logits,from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))
            if step % 100 ==0:
                print(epoch,step,'loss:',float(loss))

        total_correct,total_sum = 0,0
        for x, y in test_db:
            logits = model(x)
            prob = tf.nn.softmax(logits,axis=1)
            pred = tf.argmax(prob,axis=1)
            pred = tf.cast(pred,dtype=tf.int32)
            correct = tf.reduce_sum(tf.cast(tf.equal(pred,y),dtype=tf.int32))
            total_correct += int(correct)
            total_sum += x.shape[0]
        print(epoch,'acc:',total_correct/total_sum)

if __name__ == '__main__':
    main()
