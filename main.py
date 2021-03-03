# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
import matplotlib.image as mpimg
from tensorflow import keras
from keras.layers import Dense, Input, InputLayer, Flatten
from keras.models import Sequential, Model
from  matplotlib import pyplot as plt
from tqdm import tqdm

#%matplotlib inline

IMG_WIDTH=80
IMG_HEIGHT=80
PATH = 'C:/Users/k44sa/Documents/python/anaconda/Cuarta_entrega/data/training'

def create_dataset(img_folder):
    img_data_array =[]
    class_name=[]
    for img in tqdm(os.listdir(img_folder)):
       
        target = ''
        if img[:5] == 'chair':  
            target = 'chair'
        
        if img[:5] == 'knife':
            target = 'knife'

        if img[:8] == 'saucepan':
            target = 'saucepan'
        
        if target:
            
            image = cv2.imread(os.path.join(img_folder,img), cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            
            class_name.append(target)
            img_data_array.append(image)
        
    return img_data_array, class_name  
            

img_data, class_name = create_dataset(PATH)
        
plt.imshow(img_data[0])
    
    
target_dict = {k: v for v, k in enumerate(np.unique(class_name))}

target_val =  [target_dict[class_name[i]] for i in range(len(class_name))]


img_data_2 = np.matrix(img_data)




















 
        
        