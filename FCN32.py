from keras import models, Model
from keras.layers import *
import tensorflow as tf
from keras.applications.vgg16 import VGG16

img_w = 256
img_h = 256
n_classes = 20
partial_trainable_flag = False
total_trainable_flag = False

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(img_w, img_h,3))

if not total_trainable_flag:
    for layer in vgg16.layers:
        layer.trainable=False

elif not partial_trainable_flag:
    for i in range(0, 11):
        vgg16.layers[i].trainable=False



conv_upsampling = Conv2DTranspose(filters=n_classes, kernel_size=(32,32), strides=(32,32), padding='same', name='last_conv_fcn32', trainable=total_trainable_flag)(vgg16.layers[18].output)

reshaped_layer = Reshape((-1, n_classes))(conv_upsampling)
scores = Activation('softmax', name='fcn32_soft')(reshaped_layer)

fcn32 = Model(vgg16.input, output=scores)
#fcn32.summary()

