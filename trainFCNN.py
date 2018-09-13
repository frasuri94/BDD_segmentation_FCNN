import os
import random as rand
import numpy as np
import re
from PIL import Image
from keras import optimizers
from keras.layers import *
from keras.models import *


train_image_dir = 'bdd_dataset/train_images/'
train_label_dir = 'bdd_dataset/train_labels/'
test_image_dir = 'bdd_dataset/test_images/'
test_label_dir = 'bdd_dataset/test_labels/'

batchSize = 64
epochs = 20
num_classes = 22
train_examples_num = len(os.listdir(train_image_dir))
test_examples_num = len(os.listdir(test_image_dir))



# Shuffle the images in the folder and returns list of shuffled image names
def shuffleList(folderPath):
    files = os.listdir(folderPath)
    rand.shuffle(files)
    return files

# Loads a batch of dataset specifying indices
def getBatch(imageDir, labelDir, data_list, startIndex, endIndex):
    img_batch = []
    label_batch = []

    index = 0
    for image in data_list:
        if startIndex <= index < endIndex:
            img_batch.append(np.asarray(Image.open(imageDir+image)))
            label_batch.append(np.asarray(Image.open(labelDir+image)))
        index += 1

    img_batch = np.asarray(img_batch)
    label_batch = np.asarray(label_batch)

    return img_batch, label_batch

# Model
model = Sequential()
model.add(Conv2D(input_shape=(380, 180, 3), filters=24, kernel_size=(4, 4), strides=(3, 3), activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

# Summary
model.summary()

# Compile model
opt = optimizers.RMSprop(lr=0.001)
model.compile(loss='mse', optimizer=opt)




