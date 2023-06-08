# unet_seg_test_demo.py

### Import required library and packages
import os
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
import matplotlib.pyplot as plt
from tqdm import tqdm
import urllib
import IPython


# list all available physical devices 
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


### Test the model
""" Hyperparameters """
dataset_path = "./test/"

test_images = glob(dataset_path + "images/*")
print (len(test_images))



def download_image(url): 
    # download image from url 
    req = urllib.request.urlopen(url)
    imgarr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(imgarr, -1)
    return image 

def pre_processing(image): 
    image = cv2.resize(image, (256, 256))
    image = image/255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image 

model = tf.keras.models.load_model("./unet_road_seg.h5")
def predict(image): 
    #image = cv2.imread(path, cv2.IMREAD_COLOR)
    original_image = image.copy()
    h, w, _ = image.shape
    
    image = pre_processing(image)

    pred_mask = model.predict(image)[0]
    pred_mask = cv2.resize(pred_mask, (w, h))
    pred_mask = np.expand_dims(pred_mask, axis=-1)
    pred_mask = pred_mask > 0.5
    background_mask = np.abs(1- pred_mask)
    masked_image = original_image * pred_mask
    
    background_mask = np.concatenate([background_mask, background_mask, background_mask], axis=-1)
    background_mask = background_mask * [0, 0, 0]
    
    masked_image = masked_image + background_mask
    return masked_image


def test_local(image_paths):
    for path in tqdm(image_paths, total=len(image_paths)):
        print ('\npath -> ', path)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        masked_image = predict(image) 

        name = None
        if -1 != path.find("/"):
            name = path.split("\\")[-1]
        else:
            name = path.split("/")[-1]
        cv2.imwrite(f"./test/predict/{name}.png", masked_image)

def test_online(url): 
    image = download_image(url) 
    masked_image = predict(image) 
    cv2.imwrite(f"./test/online/predict.png", masked_image)

test_local(test_images) 

