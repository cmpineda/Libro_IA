# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 08:52:55 2024

@author: Carlos Pineda
"""

# Requisito ejecutar pip install opencv-python

import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
import numpy as np

 
image_size = 224
model = VGG19(weights='imagenet')
 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se puede acceder a la cámara")
    exit()

# Capturar una imagen
ret, frame = cap.read()
if not ret:
    print("Error: No se pudo capturar la imagen")
    exit()


# Preprocesar la imagen para el modelo VGG19
img = cv2.resize(frame, (224, 224))  # VGG19 espera imágenes de 224x224
img = img_to_array(img)  # Convertir a un arreglo numpy
img = np.expand_dims(img, axis=0)  # Añadir una dimensión extra 
img = preprocess_input(img)  # Preprocesamiento específico para VGG19

# Hacer la predicción
preds = model.predict(img)
label = decode_predictions(preds)
print("Predicción", label[0][0][1])    

cv2.putText(frame, "{}, {:.1f}".format(label[0][0][1], label[0][0][2]) , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) 
# Mostrar la imagen capturada
cv2.imshow('Imagen Capturada', frame)
# Guardar la imagen capturada (opcional)
cv2.imwrite('captura.jpg', frame)
    
key = cv2.waitKey(2000)
if key == 27:
   exit() 

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()