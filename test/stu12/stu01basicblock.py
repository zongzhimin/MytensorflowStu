import tensorflow as tf
from tensorflow.keras import layers,Sequential
# 一个网络中多个 Res Block
# 一个Res Block = 多个Basic Block

class BasicBlock(layers.Layer):
    def __init__(self,filter_num,stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num,(3,3),strides=stride,padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv2 = layers.Conv2D(filter_num,(3,3),strides=1,padding='same')
        self.bn2 = layers.BatchNormalization()
        if stride != 1:  # 保证x的维度与bn2结果一致
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num,(1,1),strides=stride))
            self.downsample.add(layers.BatchNormalization())
        else:
            self.downsample = lambda x:x

        self.stride = stride

    def call(self,inputs,training=None):
        residual = self.downsample(inputs)

        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu1 = self.relu(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        add = layers.add([bn2,residual])
        out = self.relu(add)
        return out


