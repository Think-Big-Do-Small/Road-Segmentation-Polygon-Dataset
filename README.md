# Road-Segmentation-Polygon-Dataset
Road-Segmentation With Polygon Labeled [Road Dataset](https://github.com/Think-Big-Do-Small/Road-Segmentation-Polygon-Dataset/tree/main/road_dataset) by [ImageMaskPolygonLabelAssistant (图像数据集多边形标注助手)](https://github.com/Think-Big-Do-Small/ImageMaskLabelAssistant).

### Run 
- Run program with [cuda toolkit 12.x](https://developer.nvidia.com/cuda-toolkit), [cuda-python](https://pypi.org/project/cuda-python/#history), [tensorflow-gpu](https://pypi.org/project/tensorflow-gpu/#description), opencv.

### Demo 
- road test image 1
<img width="480" height="320" src="https://github.com/Think-Big-Do-Small/Road-Segmentation-Polygon-Dataset/blob/main/road_dataset/test/images/0002.png"/>  

- road predict image 1
<img width="480" height="320" src="https://github.com/Think-Big-Do-Small/Road-Segmentation-Polygon-Dataset/blob/main/road_dataset/test/predict/0002.png.png"/>  

- road test image 2
<img width="480" height="320" src="https://github.com/Think-Big-Do-Small/Road-Segmentation-Polygon-Dataset/blob/main/road_dataset/test/images/0006.png"/>  

- road predict image 2
<img width="480" height="320" src="https://github.com/Think-Big-Do-Small/Road-Segmentation-Polygon-Dataset/blob/main/road_dataset/test/predict/0006.png.png"/>  

- road test image 3 
<img width="480" height="320" src="https://github.com/Think-Big-Do-Small/Road-Segmentation-Polygon-Dataset/blob/main/road_dataset/test/images/Road_in_Norway.jpg"/>  

- road predict image 3
<img width="480" height="320" src="https://github.com/Think-Big-Do-Small/Road-Segmentation-Polygon-Dataset/blob/main/road_dataset/test/predict/Road_in_Norway.jpg.png"/>  


### Example Code 
```bash 
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
```

### Model Weights 
- model weights download [weights](https://github.com/Think-Big-Do-Small/Road-Segmentation-Polygon-Dataset/tree/main/model)
- model train data analysis [data analysis](https://github.com/Think-Big-Do-Small/Road-Segmentation-Polygon-Dataset/blob/main/model/road_seg_data.csv)

### About Me 
- Computer Science, Master, Shenzhen University
- I am a software engineer 
- I am familar with computer languages, like c++,java,python,c,matlab,html,css,jquery
- I am familar with databases such as mysql, postgresql
- I am familar with flask, apache tomcat
- I am familar with libraries qt, opencv, caffe, keras, tensorflow, openvino
- I am familar with gpu libraries like cuda, cudnn
- I am recently doing some image segmentation projects with c++, python and cuda background matting etc. <br> 

### About Software Development Experience
- 道路分割多边形数据集 - [Road-Segmentation-Polygon-Dataset](https://github.com/Think-Big-Do-Small/Road-Segmentation-Polygon-Dataset)
- 图像数据集多边形标注助手 - [ImageMaskLabelAssistant](https://github.com/Think-Big-Do-Small/ImageMaskLabelAssistant)
- CvImageProcessingAssistant - [CvImageProcessingAssistant](https://github.com/Think-Big-Do-Small/CvImageProcessingAssistant) <br>
- Cuda-OpenCV-Object-Detection-Demo - [CvImageProcessingAssistant](https://github.com/Think-Big-Do-Small/Cuda-OpenCV-Object-Detection-Demo)<br> 
- RabbitRun(smart file packaging with high speed and efficiency)  <br> 
visit site: www.aizaozhidao.vip/tuzikuaipao 
- AI早知道(ai related projects for demostration) <br> 
visit site: www.aizaozhidao.vip 


