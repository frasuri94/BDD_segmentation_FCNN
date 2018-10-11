import os
import random as rand
import tarfile
import pickle
from keras.layers import *
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array, load_img
from SegNet import segnet
import ResNet
from Ensamble import ensamble_model
from FCN8 import fcn8
from FCN16 import fcn16
from FCN32 import fcn32


train_image_dir = 'bdd_256/train_images/'
#train_label_dir = 'bdd_dataset_resized/train_labels/'
#train_image_dir = 'bdd_dataset/train_images/'
#train_label_dir = 'bdd_dataset/train_labels/'
correct_train_labels = 'bdd_256/correct_train_labels/'
test_image_dir = 'bdd_256/test_images/'
#test_label_dir = 'bdd_dataset_resized/test_labels/'
correct_test_labels = 'bdd_256/correct_test_labels/'

batchSize = 8
epochs = 20
num_classes = 20
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
        tarName = image.replace(".jpg", ".tar.gz")
        labelName = image.replace(".jpg", "")
        if startIndex <= index < endIndex:
            img_batch.append(img_to_array(load_img(imageDir + image)))
            tarf = tarfile.open(labelDir+tarName, "r:gz")
            correct_label = pickle.load(tarf.extractfile(labelName))
            tarf.close()
            label_batch.append(correct_label)
        index += 1

    img_batch = np.array(img_batch)
    label_batch = np.array(label_batch)

    return img_batch, label_batch

#Model

#model = segnet
#model.load_weights('saved_weights/SegNet/SGD(0.1-0.9-he)/Epoch_3-TestAcc_0.7267020721435546.h5')

#model = ResNet.myResnet
#model.load_weights('saved_weights/myResnet/Adadelta+SGD(freezed+decay)/Epoch_2-TestAcc_0.7232185821533204.h5')

model = ensamble_model
model.load_weights('saved_weights/ensamble_segnet_fcn32/Epoch_6-TestAcc_0.8564886474609374.h5')

#model = fcn16
#model.load_weights('saved_weights/FCN16/Adadelta/Epoch_3-TestAcc_0.8342522125244141.h5')

# Summary
model.summary()

# Compile model
opt = SGD(lr=0.0001)
#opt = 'adadelta'
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Training

def train_model():

    for epoch in range(0, epochs):

        print("Epoch ---> " + str(epoch + 1) + "/" + str(epochs))

        train_list = shuffleList(train_image_dir)

        startIndex = 0
        endIndex = batchSize
        cont_loss = 0.0
        cont_acc = 0.0
        num_batches_train = int(train_examples_num/batchSize)


        for batch in range(0, num_batches_train):

            img_batch, label_batch = getBatch(train_image_dir, correct_train_labels, train_list, startIndex, endIndex)

            #metrics = model.train_on_batch(img_batch, label_batch)
            metrics = model.train_on_batch([img_batch, img_batch], label_batch)
            batch_loss = metrics[0]
            batch_accuracy = metrics[1]
            cont_loss+=batch_loss
            cont_acc+=batch_accuracy

            startIndex = endIndex
            endIndex = startIndex + batchSize

            print("Epoch: " + str(epoch + 1) + "/" + str(epochs)
                  + "    Batch " + str(batch + 1) + "/" + str(num_batches_train)
                  + "    Batch loss: " + str(batch_loss)
                  + "    Avg loss: " + str(cont_loss/(batch + 1))
                  + "    Batch accuracy: " + str(batch_accuracy)
                  + "    Avg accuracy: " + str(cont_acc/(batch + 1)))


        print("Testing the model...")

        test_list = shuffleList(test_image_dir)
        startIndex = 0
        endIndex = batchSize
        cont_loss = 0.0
        cont_acc = 0.0
        num_batches_test = int(test_examples_num / batchSize)


        for batch in range(0, num_batches_test):

            img_batch, label_batch = getBatch(test_image_dir, correct_test_labels, test_list, startIndex, endIndex)

            #metrics = model.test_on_batch(img_batch, label_batch)
            metrics = model.test_on_batch([img_batch, img_batch], label_batch)
            batch_loss = metrics[0]
            batch_accuracy = metrics[1]
            cont_loss += batch_loss
            cont_acc += batch_accuracy

            startIndex = endIndex
            endIndex = startIndex + batchSize

            print("Epoch: " + str(epoch + 1) + "/" + str(epochs)
                  + "    Batch " + str(batch + 1) + "/" + str(num_batches_test)
                  + "    Batch loss: " + str(batch_loss)
                  + "    Avg loss: " + str(cont_loss / (batch + 1))
                  + "    Batch accuracy: " + str(batch_accuracy)
                  + "    Avg accuracy: " + str(cont_acc / (batch + 1)))

        model.save_weights('saved_weights/ensamble_segnet_fcn32/SGD/Epoch_' + str(epoch + 1) + '-TestAcc_' + str(cont_acc / (batch + 1)) + '.h5', overwrite=False)

train_model()