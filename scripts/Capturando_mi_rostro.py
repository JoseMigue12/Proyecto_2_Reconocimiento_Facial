import cv2
import os

#imagesPath = "C:/Users\heloj/Documents/Redes Neuronales/Proyecto_rostros/Mi_rostro2"
imagesPath = "C:/Users\heloj/Documents/Redes Neuronales/Proyecto_rostros/Data_procesamiento/Mi_rostro2"
imagesPathList = os.listdir(imagesPath)

if not os.path.exists('Mi rostro encontrado3'):
    print('Carpeta creada: Mi rostro encontrado3')
    os.makedirs('Mi rostro encontrado3')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

count = 0
for imageName in imagesPathList:
    image = cv2.imread(imagesPath+'/'+imageName)
    imageAux = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = faceClassif.detectMultiScale(gray, 2.1, 1)

    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y),(x+w,y+h),(128,0,255),2)
    cv2.rectangle(image,(10,5),(450,25),(255,255,255),-1)

    for (x,y,w,h) in faces:
            rostro = imageAux[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
            #cv2.imshow('rostro',rostro)
            cv2.waitKey(0)
            cv2.imwrite('Mi rostro encontrado3/mirostro_{}.jpg'.format(count),rostro)
            count = count +1

cv2.destroyAllWindows()

