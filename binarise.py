# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 08:49:18 2021

@author: k44sa
"""

import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

from keras.utils import to_categorical
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Sequential


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

SIZE = 80

PATH = 'C:/Users/k44sa/Documents/python/anaconda/Cuarta_entrega/data/testing/'
dataset = []  # Dataset  
label = []  # 0 chait : 1 knife



def create_dataset(img_folder,dataset_img,label_class):
    
    for img in tqdm(os.listdir(img_folder)):
   
        target = -1
        if img[:5] == 'chair':  
 
            target = 0
        
        if img[:5] == 'knife':
     
            target = 1
        if img[:8] == 'saucepan':
          
            target = 2
        if target != -1:
          
            try:   
        
                #CARGAMOS LA IMAGEN
                image = cv2.imread(img_folder+img)
                #CAMBIAMOS EL FORMATO A RGB
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                #SUAVIZAMOS LA IMAGEN
                image = cv2.GaussianBlur(image,(3,3),0)
                #CONVERTIOMS A ESCALA DE GRICES
                image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                #BINARIZAMOS
                ret,image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
                #CAMBIAMOS EL TAMAÑO
                new_array = cv2.resize(image, (SIZE, SIZE)) 
                #AGREGAMOS AL DATASET
                dataset_img.append(new_array) 
                label_class.append(target)
            except Exception:
                pass 
           
#############################################################################
#                                   BASE DE DATOS
#############################################################################                
create_dataset(PATH,dataset,label)                  


df = np.array(dataset).reshape(-1, SIZE, SIZE)
df = df.reshape(df.shape[0], df.shape[1], df.shape[2], 1)


#plt.imshow(df[200],cmap='gray')

#dtf = df.reshape((len(df),np.prod(df.shape[1:])))

#dtf = dtf.astype('float32')
#dtf /= 255


X_train, X_test, y_train, y_test = train_test_split(df,to_categorical(np.array(label)), test_size = 0.25, random_state = 0)


#############################################################################
#                        ARQUITECTURA DEL MODELO
############################################################################# 
model = None
model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(SIZE,SIZE,1), activation = 'relu', data_format='channels_last'))
model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))
model.add(Convolution2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(activation = 'relu', units=512))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))
model.add(Dense(activation = 'relu', units=256))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))
model.add(Dense(activation = 'sigmoid', units=3))
#model.add(Dense(activation = 'softmax', units=3))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())

#############################################################################
#                                   ENTRENAMIENTO
#############################################################################  
history = model.fit(np.array(X_train),
                    y_train, 
                    batch_size = 64, 
                    verbose = 1, 
                    epochs = 20,  
                    validation_split = 0.1,
                    shuffle = False)






#############################################################################
#                           RESULTADOS DE ENTRENAMEINTO
#############################################################################

print("Test Accuracy: {:.2f}%".format(model.evaluate(np.array(X_test), np.array(y_test))[1]*100))

plt.plot(history.history['val_loss'])
plt.plot(history.history['val_accuracy'])

#target_dict = {k: v for v, k in enumerate(np.unique(label))}




prediction = model.predict(np.array(X_test))
y_test_matrix = [np.argmax(t) for t in y_test]
y_prediction_matrix = [np.argmax(t) for t in prediction]
confusion = confusion_matrix(y_test_matrix,y_prediction_matrix)

sns.heatmap(confusion,annot=True,fmt="d",cbar=False)
plt.show()


x = 155
plt.imshow(X_test[x],cmap='gray')

pred_individual = model.predict(np.expand_dims(X_test[x],axis = 0))
print(pred_individual)
np.round(pred_individual,2)

