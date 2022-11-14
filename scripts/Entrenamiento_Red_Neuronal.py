import numpy as np
from skimage.io import imread_collection , concatenate_images
import matplotlib.pyplot as plt
import copy  as cp
import copy
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Flatten,Dropout
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

folderMiguel = 'Data/Miguel/*.jpg'
folderOthers = 'Data/Desconocido/*.jpg'

# Cargando datos 
imagesMiguel = imread_collection(folderMiguel)
imagesOthers = imread_collection(folderOthers) 

#Cantidad de imagenes
nMiguel  = len(imagesMiguel)
nOthers = len(imagesOthers)

# union de los datos
images = np.append(imagesMiguel, imagesOthers, axis=0)

print("Total de imagenes: ",len(images))

plt.imshow(images[0])
print(images[0].shape)

def Create_Y():
     return [0]*nMiguel + [1]*nOthers
Y = Create_Y()

Y = np.array(Y)
X = np.array(images)

from skimage.transform import resize
X=resize(X,(len(images),64,64,3))

plt.imshow(X[0])
print(X[0].shape)

modelo=Sequential() 

modelo.add(Conv2D(200,(3,3),input_shape=X.shape[1:]))
modelo.add(Activation('relu'))
modelo.add(MaxPooling2D(pool_size=(2,2)))

modelo.add(Conv2D(100,(3,3)))
modelo.add(Activation('relu'))
modelo.add(MaxPooling2D(pool_size=(2,2)))

modelo.add(Conv2D(50,(3,3)))
modelo.add(Activation('relu'))
modelo.add(MaxPooling2D(pool_size=(2,2)))

modelo.add(Flatten()) 
modelo.add(Dropout(0.5)) 

modelo.add(Dense(50,activation='relu'))
modelo.add(Dense(2,activation='softmax')) 

modelo.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

Y=to_categorical(Y)

Y[0]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0)

#Guardamos nuestro modelo entrenado en una carpeta
checkpoint = ModelCheckpoint('CNN/model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

history=modelo.fit(X_train,Y_train,epochs=20,callbacks=[checkpoint],validation_split=0.2) #,shuffle = True)

