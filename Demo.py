import keras
import numpy as np
from keras_preprocessing import image
from keras.preprocessing.image import img_to_array, load_img, array_to_img
import os
from SegNet import segnet
from FCN32 import fcn32

classes = np.array([
    [107., 142., 35.],      #vegetation
    [0., 0., 0.],           #background/unknown
    [70., 70., 70.],	    #building
    [0., 0., 142.],	        #car
    [0., 0., 70.],		    #truck
    [128., 64., 128.],	    #road
    [220., 220., 0.],	    #sign
    [153., 153., 153.],	    #pole
    [250., 170., 30.],	    #traffic light
    [70., 130., 180.],	    #sky
    [244., 35., 232.],	    #sidewalk
    [190., 153., 153.],	    #fence
    [220., 20., 60.],	    #person
    [152., 251., 152],	    #terrain
    [119., 11., 32.],	    #bike
    [0., 60., 100.]	,       #bus
    [102., 102., 156.],	    #wall
    [0., 80., 100.],        #caravan
    [255., 0., 0.],	        #rider
    [0., 0., 230.]          #motorcycle
])

def pred_array_to_seg_array(pred_array):
    seg_array = np.zeros((256 * 256, 3))
    rows, cols = pred_array.shape

    max_indices = np.argmax(pred_array, axis=1)
    for i in range(0, rows):
        seg_array[i] = classes[max_indices[i]]

    seg_array = np.reshape(seg_array, (256, 256, 3))
    return seg_array

model = fcn32
fcn32.load_weights('saved_weights/FCN32/Epoch_7-TestAcc_0.852270248413086.h5')

img_path = 'image.jpg'
img = image.load_img(img_path, target_size=(256, 256))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = model.predict(x)
pred_array = preds[0]
seg_array = pred_array_to_seg_array(pred_array)
seg_image = array_to_img(seg_array)
seg_image.show()
