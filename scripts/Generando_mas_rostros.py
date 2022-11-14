from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import io

datagen = ImageDataGenerator(
        rotation_range=35,     #Random rotation between 0 and 45
        width_shift_range=0.1,   #% shift
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='constant', cval=125)    #Also try nearest, constant, reflect, wrap


x = io.imread('C:/Users/heloj/Documents/Redes Neuronales/Proyecto_rostros/Mi rostro encontrado/mi_rostro (146).jpg') 
x = x.reshape((1, ) + x.shape)  #Array with shape (1, 256, 256, 3)

i = 0
for batch in datagen.flow(x, batch_size=16,  
                          save_to_dir='Miguel', 
                          save_prefix='mrostro1', 
                          save_format='jpg'):
    i += 1
    if i > 12:
        break  # otherwise the generator would loop indefinitely  
