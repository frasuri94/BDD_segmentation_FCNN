from keras import models, Model
from keras.layers.core import Activation, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization

img_w = 256
img_h = 256
n_classes = 20
kernel = 3
block1_filters = 8
block2_filters = block1_filters * 2
block3_filters = block2_filters * 2
block4_filters = block3_filters * 2
block5_filters = block4_filters
trainable_flag = False
kernel_init = 'he_normal'

encoding_layers = [
    Conv2D(block1_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init, input_shape=(img_w, img_h,3)),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),
    Conv2D(block1_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),
    MaxPooling2D(trainable=trainable_flag),

    Conv2D(block2_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),
    Conv2D(block2_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),
    MaxPooling2D(trainable=trainable_flag),

    Conv2D(block3_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),
    Conv2D(block3_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),
    Conv2D(block3_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),
    MaxPooling2D(trainable=trainable_flag),

    Conv2D(block4_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),
    Conv2D(block4_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),
    Conv2D(block4_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),
    MaxPooling2D(trainable=trainable_flag),

    Conv2D(block5_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),
    Conv2D(block5_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),
    Conv2D(block5_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),
    MaxPooling2D(trainable=trainable_flag),
]

model = models.Sequential()
model.encoding_layers = encoding_layers

for l in model.encoding_layers:
    model.add(l)

decoding_layers = [
    UpSampling2D(),
    Conv2D(block5_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),
    Conv2D(block5_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),
    Conv2D(block5_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),

    UpSampling2D(),
    Conv2D(block4_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),
    Conv2D(block4_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),
    Conv2D(block3_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),

    UpSampling2D(),
    Conv2D(block3_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),
    Conv2D(block3_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),
    Conv2D(block2_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),

    UpSampling2D(),
    Conv2D(block2_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),
    Conv2D(block1_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),

    UpSampling2D(),
    Conv2D(block1_filters, kernel, kernel, border_mode='same', trainable=trainable_flag, kernel_initializer=kernel_init),
    BatchNormalization(trainable=trainable_flag),
    Activation('relu', trainable=trainable_flag),
    Conv2D(n_classes, 1, 1, border_mode='valid', trainable=trainable_flag, kernel_initializer=kernel_init, name='last_conv_segnet'),
    BatchNormalization(trainable=trainable_flag),
]
model.decoding_layers = decoding_layers
for l in model.decoding_layers:
    model.add(l)

model.add(Reshape((img_h * img_w, n_classes), trainable=trainable_flag))
model.add(Activation('softmax', trainable=trainable_flag, name='seg_soft'))
segnet = Model(inputs=model.input, outputs=model.get_layer('seg_soft').output)
#segnet.summary()