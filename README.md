# Proyecto_2_Reconocimiento_Facial
En este repositorio se subió el Proyecto 2 que trata sobre reconocimiento facial con una red neuronal convolucional. 

# Reporte.

## Estrategia general.

Se utilizo un subconjunto de la base de datos CelabA, el subconjunto es de 16000 imágenes y otro conjunto de mi rostro que lo conforman 200 imágenes.

En el subconjunto de CelabA se recortaron las imágenes con finalidad de solo tener el rostro. Esto lo hacemos con la ayuda de las paqueterías cv2 y os. Utilizamos cv2 ya que cuenta con el reconocimiento de rostros, primeramente, nos posicionamos en la carpeta donde están guardadas las imágenes para posteriormente crear una carpeta donde guardara los rostros y con la ayuda de un ciclo for pasamos por todas las imágenes originales de CelabA en la carpeta donde las tenemos y se recortan.

Para mi rostro se hace el mismo proceso que con el subconjunto de CelabA.

Los procedimientos anteriores se realizan en los Notebooks Capturando_rostros y Capturando_mi_rostro.
Como no cuento con un amplio conjunto de datos de mi rostro se utilizó de la paquetería de Tensorflow la función ImageDataGenerator para generar más imágenes a partir de los rostros que ya tenemos, así nuestro conjunto de datos de mi rostro creció a 2000 imágenes, esto se realiza en el Notebook Generando más rostros.

Así el total de mis datos son 18000 imágenes de rostros.

Después se etiquetan las imágenes, las imágenes de mi rostro se etiquetan con 0 y los rostros de las personas desconocidas con 1. Ya teniendo los datos listos se pasó a realizar y entrenar una Red Neuronal convolucional. 

## Problemas que se encontraron.

Uno de los problemas que se tuvieron al principio fue por tema de memoria, me salía un error respecto a esto mencionaba que no podía generar un vector de tamaño grande entonces por eso se decidió trabajar con un subconjunto de CelabA.

Otro de los problemas que se tuvo fue que al recortar las imágenes para obtener solo los rostros, se generaban algunas nuevas imágenes que no correspondían a rostros y también rostros a la mitad. Esto se resolvió eliminando manualmente.

## Descripción de la Red Neuronal.

Se utilizo una Red Neuronal convolucional para reconocimiento facial con la ayuda de la paquetería TensorFlow. La estructura de la red neuronal es de un modelo Secuencial es decir varias capas apiladas entre ellas. Nuestra red cuenta con 3 capas convolucionales y dos capas densas. Las capas convolucionales cuentan con una función de activación Relu y un MaxPooling2D.

Posteriormente de las capas convolucionales con add(Flatten()) hacemos que la Imagen profunda se vuelva plana, es decir solo de una dimensión.

Después tenemos dos capas Densas, la primera cuanta con 50 neuronas y una función de activación Relu, la siguiente capa densa que es la capa final cuenta con 2 neuronas o dos salidas para dos categorías y ocupa una función de activación Softmax.

Para nuestra Red Neuronal se ocupa una función de costo crossentropy y el optimizador que se utiliza es el Adam.

## Captura de Pantalla del Entrenamiento.

## Conclusiones.

A partir de la experiencia del proyecto, me parece muy interesante saber que las redes neuronales son una de las herramientas importantes y muy utilizadas por el hombre para diversas cosas como en el caso del reconocimiento facial. El proyecto tuvo muy buenos resultados ya que en la época 19 tuvimos una precisión de 0.9976 y un valor de la función de costo de 0.0069, en las gráficas siguientes podemos ver como se comportaron estos por el paso de las épocas tanto en los datos de entrenamiento y prueba.


Gracias a las graficas podemos ver que nuestra red neuronal fue buena y no hay presencia de sobreajuste.
