from keras import models, Model
from keras.layers import *
from keras.layers.core import *
import tensorflow as tf
from keras.applications.resnet50 import ResNet50

img_w = 256
img_h = 256
n_classes = 20
trainable_flag = False

resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(img_w, img_h, 3))

if not trainable_flag:
    for l in resnet.layers:
        l.trainable= False

up1 = Conv2DTranspose(filters=256, kernel_size=4, strides=4, trainable=trainable_flag)(resnet.output)
norm1 = BatchNormalization(trainable=trainable_flag)(up1)
act1 = Activation('relu', trainable=trainable_flag)(norm1)

up2 = Conv2DTranspose(filters=64, kernel_size=4, strides=4, trainable=trainable_flag)(act1)
norm2 = BatchNormalization(trainable=trainable_flag)(up2)
act2 = Activation('relu', trainable=trainable_flag)(norm2)

up3 = Conv2DTranspose(filters=n_classes, kernel_size=2, strides=2, trainable=trainable_flag, name='last_conv_resnet')(act2)
norm3 = BatchNormalization(trainable=trainable_flag)(up3)

reshape = Reshape((-1, n_classes), trainable=trainable_flag)(norm3)
scores = Activation('softmax', trainable=trainable_flag)(reshape)

myResnet = Model(inputs=resnet.input, outputs=scores)
#myResnet.summary()