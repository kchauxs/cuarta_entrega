# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:57:23 2021

@author: k44sa
"""

import os 
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from tensorflow import keras
from keras.layers import Dense, Input, InputLayer, Flatten,Dropout
from keras.models import Sequential, Model
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



IMG_SIZE = 90
PATH = 'C:/Users/k44sa/Documents/python/anaconda/Cuarta_entrega/data/testing'
PATH_NPY = 'C:/Users/k44sa/Documents/python/anaconda/Cuarta_entrega/data/'



def create_dataset(img_folder):
    training_data = []
    class_name = []
    for img in tqdm(os.listdir(img_folder)):
       
        target = 0
        if img[:5] == 'chair':  
            #target = 'chair'
            target = 1
        
        if img[:5] == 'knife':
            #target = 'knife'
            target = 2
        if img[:8] == 'saucepan':
            #target = 'saucepan'
            target = 3
        if target != 0:
          
            try:
                img_array = cv2.imread(os.path.join(PATH,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array])  # add this to our training_data
                class_name.append(target)
            except Exception:
                pass          
            
    return training_data, class_name  


training_data , class_name = create_dataset(PATH)

X = []

for features in training_data:
    X.append(features)

#print(X[0].reshape(-1, IMG_WIDTH, IMG_WIDTH))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE)

imgagen_test = X[200]
plt.imshow(imgagen_test, cmap="gray")
imgagen_test.shape


#np.save(os.path.join(PATH_NPY,'features'),np.array(img_data))
#df = np.load(os.path.join(PATH_NPY,'features.npy'))

df = X

df = df.reshape((len(df),np.prod(df.shape[1:])))


df = df.astype('float32')
df /= 255


target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
target_val =  [target_dict[class_name[i]] for i in range(len(class_name))]



x_entrenamiento,x_prueba,y_entrenamiento,y_prueba = train_test_split(df,target_val,test_size = 0.3, random_state = 0)


y_entrenamiento = np_utils.to_categorical(y_entrenamiento,3)
y_prueba = np_utils.to_categorical(y_prueba,3)


modelo = Sequential()
modelo.add(Dense(units = 640,activation='relu',input_dim= IMG_SIZE*IMG_SIZE ))
modelo.add(Dropout(0.2))
modelo.add(Dense(units = 320,activation='relu' ))
modelo.add(Dropout(0.2))
modelo.add(Dense(units = 100,activation='relu' ))
modelo.add(Dropout(0.2))
modelo.add(Dense(units = 3,activation='softmax' ))
 

modelo.summary()


modelo.compile(optimizer='adam',loss ='categorical_crossentropy',metrics=['accuracy'])

historico = modelo.fit(x_entrenamiento,y_entrenamiento,epochs=40,validation_data=(x_prueba,y_prueba))

historico.history.keys()
plt.plot(historico.history['val_loss'])
plt.plot(historico.history['val_accuracy'])

prediccion = modelo.predict(x_prueba)

y_prueba_matriz = [np.argmax(t) for t in y_prueba]
y_prediccion_matriz = [np.argmax(t) for t in prediccion]
confusion = confusion_matrix(y_prueba_matriz,y_prediccion_matriz)




