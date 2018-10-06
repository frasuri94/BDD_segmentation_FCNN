from keras import models, Model
from keras.layers import *
import tensorflow as tf
from keras.applications.vgg16 import VGG16

img_w = 256
img_h = 256
n_classes = 20
trainable_flag = False

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(img_w, img_h,3))

if not trainable_flag:
    for i in range(0, 11):
        vgg16.layers[i].trainable=False

pool4 = vgg16.layers[14].output
pool3 = vgg16.layers[10].output

conv_upsampling_1 = Conv2DTranspose(filters=512, kernel_size=(2,2), strides=(2,2), padding='same')(vgg16.layers[18].output)
skip_connection_1 = Add()([conv_upsampling_1, pool4])
conv_upsampling_2 = Conv2DTranspose(filters=256, kernel_size=(2,2), strides=(2,2), padding='same')(skip_connection_1)
skip_connection_2 = Add()([conv_upsampling_2, pool3])
conv_upsampling_3 = Conv2DTranspose(filters=n_classes, kernel_size=(8,8), strides=(8,8), padding='same')(skip_connection_2)

reshaped_layer = Reshape((-1, n_classes))(conv_upsampling_3)
scores = Activation('softmax')(reshaped_layer)

fcn8 = Model(vgg16.input, output=scores)
#fcn8.summary()

