from SegNet import segnet
from FCN32 import fcn32
from keras import Model
from keras.layers import Add, Average, BatchNormalization, Reshape, Activation

img_w = 256
img_h = 256
n_classes = 20

segnet.load_weights('saved_weights/SegNet/SGD(decay)/Epoch_3-TestAcc_0.8185689849853516.h5')
fcn32.load_weights('saved_weights/FCN32/Epoch_7-TestAcc_0.852270248413086.h5')

#Ensamble with last convolutionals
last_conv_segnet = segnet.get_layer('last_conv_segnet').output
last_conv_fcn32 = fcn32.get_layer('last_conv_fcn32').output
combined_conv = Add()([last_conv_segnet, last_conv_fcn32])
last = BatchNormalization()(combined_conv)


reshaped = Reshape((img_h * img_w, n_classes))(last)
scores = Activation('softmax')(reshaped)

ensamble_model = Model(inputs=[segnet.input, fcn32.input], outputs=scores)
#ensamble_model.summary()