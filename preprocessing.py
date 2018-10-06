import os
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from PIL import Image
import random as rand
import numpy as np
import pickle
import tarfile
from PIL import Image

imgDir = 'bdd_dataset/train_images/'
labelDir = 'bdd_dataset/train_labels/'
#imgDir = 'bdd_dataset/test_images/'
#labelDir = 'bdd_dataset/test_labels/'
newImgDir = 'bdd_256/train_images/'
#newLabelDir = 'bdd_256/train_labels/'
#newImgDir = 'bdd_256/test_images/'
newLabelDir = 'bdd_256/test_labels/'
#correct_labels_dir = 'bdd_256/correct_train_labels/'
correct_labels_dir = 'bdd_256/correct_test_labels/'

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
    [0., 0., 230.]
])


def resizeImages():
    imgs = os.listdir(imgDir)
    labels = os.listdir(labelDir)

    for img in imgs:
        original_image = load_img(imgDir + img)
        resized_image = original_image.resize((456, 256))
        cropped_image = resized_image.crop((100,0, 356,256))
        cropped_image.save(newImgDir + img)

    for label in labels:
        original_label = load_img(labelDir+label).convert('RGB')
        resized_label = original_label.resize((456,256), Image.NEAREST)
        cropped_label = resized_label.crop((100, 0, 356, 256))
        modified_name = (label.replace("_train_color", ""))
        cropped_label.save(newLabelDir+modified_name)


#resizeImages()


def transformLabel(labelDir):
    list = os.listdir(labelDir)
    #index = 1

    for label in list:
        newLabel = np.zeros((256 * 256, 20))
        tarName = label.replace('.png', '.tar.gz')
        if not (os.path.exists(correct_labels_dir+tarName)):
            l = load_img(labelDir+label)
            l = np.reshape(img_to_array(l, data_format="channels_last"), (-1, 3))
            r, c = l.shape

            for i in range(0, r):
                for k in range(0,20):
                    if(l[i,0]==classes[k,0] and l[i,1]==classes[k,1] and l[i,2]==classes[k,2]):
                        newLabel[i, k] = 1
                        break

            pureLabelName = label.replace('.png', '')
            with open(correct_labels_dir+pureLabelName, 'wb') as f:
                pickle.dump(newLabel, f)

            tar = tarfile.open(correct_labels_dir+tarName, "w:gz")
            tar.add(correct_labels_dir+pureLabelName, pureLabelName)
            tar.close()
            os.remove(correct_labels_dir+pureLabelName)
        #print(index)
        #index+=1


#transformLabel(newLabelDir)