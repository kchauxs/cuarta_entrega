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
from keras.models import load_model
from keras.utils import to_categorical
from PIL import Image

IMG_SIZE = 90
PATH = 'C:/Users/k44sa/Documents/python/anaconda/Cuarta_entrega/data/testing'
PATH_NPY = 'C:/Users/k44sa/Documents/python/anaconda/Cuarta_entrega/data/'



def create_dataset(img_folder):
    training_data = []
    class_name = []
    for img in tqdm(os.listdir(img_folder)):
       
        target = 0
        if img[:5] == 'chair':  
            target = 'chair'
            #target = 1
        
        if img[:5] == 'knife':
            target = 'knife'
            #target = 2
        if img[:8] == 'saucepan':
            target = 'saucepan'
            #target = 3
        if target != 0:
          
            try:
                #cargamos imagen
                img_array = cv2.imread(os.path.join(PATH,img))
                
                #convertimos imagen a formato RGB
                img_array = cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)
                
                #Creamos imagen a escala de grises
                #img_array = cv2.cvtColor(img_array,cv2.COLOR_RGB2GRAY)
                
                #Cambiamos su tamanio
                new_array =  
                
                #agregamos a nuestro base de datos de entrenamiento.
                training_data.append([new_array])
                
                #agregamos su clase
                class_name.append(target)
                
                #img_array = cv2.imread(os.path.join(PATH,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                #new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                #training_data.append([new_array])  # add this to our training_data
                #class_name.append(target)
            except Exception:
                pass          
            
    return training_data, class_name  


training_data , class_name = create_dataset(PATH)



#############################################################################
image_directory = 'C:/Users/k44sa/Documents/python/anaconda/Cuarta_entrega/data/testing/'
SIZE = 64
dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
label = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

#parasitized_images = os.listdir(image_directory)
#for i, image_name in tqdm( enumerate(parasitized_images)):    #Remember enumerate method adds a counter and returns the enumerate object
    
#    if (image_name.split('.')[1] == 'jpg'):
#        image = cv2.imread(image_directory  + image_name)
#        image = Image.fromarray(image, 'RGB')
#        image = image.resize((SIZE, SIZE))
#        dataset.append(np.array(image))
#        label.append(0)




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
            
create_dataset(image_directory)




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


target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
target_val =  [target_dict[class_name[i]] for i in range(len(class_name))]

X_train, X_test, y_train, y_test = train_test_split(dataset,to_categorical(np.array(label)), test_size = 0.20, random_state = 0)





history = model.fit(np.array(X_train), 
                         y_train, 
                         batch_size = 64, 
                         verbose = 1, 
                         epochs = 5,      #Changed to 3 from 50 for testing purposes.
                         validation_split = 0.1,
                         shuffle = False
                      #   callbacks=callbacks
                     )

# ## Accuracy calculation
# 
# I'll now calculate the accuracy on the test data.

print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(X_test), np.array(y_test))[1]*100))

history.history.keys()
plt.plot(history.history['val_loss'])
plt.plot(history.history['val_accuracy'])










#############################################################################




#X = []

#for features in training_data:
    #X.append(features)

#print(X[0].reshape(-1, IMG_WIDTH, IMG_WIDTH))
#X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE)



training_data = np.array(training_data).reshape(-1, IMG_SIZE, IMG_SIZE)

imgagen_test = training_data[70]
plt.imshow(imgagen_test, cmap="gray")
imgagen_test.shape


#np.save(os.path.join(PATH_NPY,'features'),np.array(img_data))
#df = np.load(os.path.join(PATH_NPY,'features.npy'))

df = training_data

df = df.reshape((len(df),np.prod(df.shape[1:])))


df = df.astype('float32')
df /= 255


target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
target_val =  [target_dict[class_name[i]] for i in range(len(class_name))]



x_entrenamiento,x_prueba,y_entrenamiento,y_prueba = train_test_split(df,target_val,test_size = 0.3, random_state = 0)




y_entrenamiento = np_utils.to_categorical(y_entrenamiento,3)
y_prueba = np_utils.to_categorical(y_prueba,3)


modelo = Sequential()
modelo.add(Dense(units = 810,activation='relu',input_dim= IMG_SIZE*IMG_SIZE ))
modelo.add(Dropout(0.2))
modelo.add(Dense(units = 410,activation='relu' ))
modelo.add(Dropout(0.2))
modelo.add(Dense(units = 210,activation='relu' ))
modelo.add(Dropout(0.2))
modelo.add(Dense(units = 110,activation='relu' ))
modelo.add(Dropout(0.2))
modelo.add(Dense(units = 3,activation='softmax' ))
 

modelo.summary()


modelo.compile(optimizer='adam',loss ='categorical_crossentropy',metrics=['accuracy'])

historico = modelo.fit(
    x_entrenamiento,
    y_entrenamiento,
    epochs=40,
    validation_data=(x_prueba,y_prueba)
    )

historico.history.keys()
plt.plot(historico.history['val_loss'])
plt.plot(historico.history['val_accuracy'])

prediccion = modelo.predict(x_prueba)

y_prueba_matriz = [np.argmax(t) for t in y_prueba]
y_prediccion_matriz = [np.argmax(t) for t in prediccion]
confusion = confusion_matrix(y_prueba_matriz,y_prediccion_matriz)

## GUARDAR MODELO
modelo.save('intento1.h5')


#########################################################################
## CARGAR MODELO PREVIAMENTE ENTRENADO
#########################################################################


nevo_modelo = load_model('intento1.h5')
historico = nevo_modelo.fit(
    x_entrenamiento,
    y_entrenamiento,
    epochs=40,
    validation_data=(x_prueba,y_prueba)
    )

historico.history.keys()
plt.plot(historico.history['val_loss'])
plt.plot(historico.history['val_accuracy'])

prediccion = modelo.predict(x_prueba)
y_prueba_matriz = [np.argmax(t) for t in y_prueba]
y_prediccion_matriz = [np.argmax(t) for t in prediccion]
confusion = confusion_matrix(y_prueba_matriz,y_prediccion_matriz)


## GUARDAR MODELO
modelo.save('intento1.h5')










