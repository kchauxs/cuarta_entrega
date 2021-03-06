# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 19:21:53 2021

@author: k44sa
"""
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from PIL import Image

image_directory = 'C:/Users/k44sa/Documents/python/anaconda/Cuarta_entrega/data/testing/'
SIZE = 64
dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
label = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.


def create_dataset(img_folder):
    for img in tqdm(os.listdir(img_folder)):
       
        target = -1
        if img[:5] == 'chair':  
            #target = 'chair'
            target = 0
        
        if img[:5] == 'knife':
            #target = 'knife'
            target = 1
        #if img[:8] == 'saucepan':
            #target = 'saucepan'
            #target = 2
        if target != -1:
          
            try:
               
                image = cv2.imread(img_folder  + img)
                image = Image.fromarray(image, 'RGB')
                image = image.resize((SIZE, SIZE))
                dataset.append(np.array(image))
                label.append(target)
           
            except Exception:
                pass     
            
#############################################################################
#                                   MODELO
#############################################################################            
INPUT_SHAPE = (SIZE, SIZE, 3)   #change to (SIZE, SIZE, 3)
inp = keras.layers.Input(shape=INPUT_SHAPE)

conv1 = keras.layers.Conv2D(32, kernel_size=(3, 3), 
                               activation='relu', padding='same')(inp)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
norm1 = keras.layers.BatchNormalization(axis = -1)(pool1)
drop1 = keras.layers.Dropout(rate=0.2)(norm1)
conv2 = keras.layers.Conv2D(32, kernel_size=(3, 3), 
                               activation='relu', padding='same')(drop1)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
norm2 = keras.layers.BatchNormalization(axis = -1)(pool2)
drop2 = keras.layers.Dropout(rate=0.2)(norm2)

flat = keras.layers.Flatten()(drop2)  #Flatten the matrix to get it ready for dense.

hidden1 = keras.layers.Dense(512, activation='relu')(flat)
norm3 = keras.layers.BatchNormalization(axis = -1)(hidden1)
drop3 = keras.layers.Dropout(rate=0.2)(norm3)
hidden2 = keras.layers.Dense(256, activation='relu')(drop3)
norm4 = keras.layers.BatchNormalization(axis = -1)(hidden2)
drop4 = keras.layers.Dropout(rate=0.2)(norm4)

out = keras.layers.Dense(2, activation='sigmoid')(drop4)   #units=1 gives error

model = keras.Model(inputs=inp, outputs=out)
model.compile(optimizer='adam',
                loss='categorical_crossentropy',   #Check between binary_crossentropy and categorical_crossentropy
                metrics=['accuracy'])
print(model.summary())
         
            
#############################################################################
#                                   ENTRENAMIENTO
#############################################################################         
            
create_dataset(image_directory)

X_train, X_test, y_train, y_test = train_test_split(dataset,to_categorical(np.array(label)), test_size = 0.20, random_state = 0)

history = model.fit(np.array(X_train), 
                         y_train, 
                         batch_size = 64, 
                         verbose = 1, 
                         epochs = 15,      #Changed to 3 from 50 for testing purposes.
                         validation_split = 0.1,
                         shuffle = False
                      #   callbacks=callbacks
                     )



print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(X_test), np.array(y_test))[1]*100))

history.history.keys()
plt.plot(history.history['val_loss'])
plt.plot(history.history['val_accuracy'])


y_train[100]
new1 = X_train[100]

plt.imshow(new1)

new2 = np.expand_dims(new1,axis = 0)
pred_individual = model.predict(new2)














