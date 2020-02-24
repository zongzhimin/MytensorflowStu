import tensorflow as tf

from tensorflow.keras import optimizers

optimizer = optimizers.SGD(learning_rate=0.2, momentum=0.9)

for epoch in range(100):
    # get loss

    # change learning rate
    optimizer.learning_rate = 0.2 * (100 - epoch) / 100

    # update weights
