import os
from PIL import Image

imgDir = 'bdd_dataset/test_images/'
labelDir = 'bdd_dataset/test_labels/'
newImgDir = 'bdd_dataset_resized/test_images/'
newLabelDir = 'bdd_dataset_resized/test_labels/'

def resizeImages(imgDir):
    list = os.listdir(imgDir)

    for img in list:
        original_image = Image.open(imgDir+img)
        resized_image = original_image.resize((320,180), Image.ANTIALIAS)
        modified_name = (img.replace("_train_color", ""))
        resized_image.save(newLabelDir+modified_name)

resizeImages(labelDir)