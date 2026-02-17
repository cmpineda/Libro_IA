# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 08:52:55 2024

@author: Carlos Pineda
"""

# Requisito ejecutar pip install opencv-python

import cv2
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

 
image_size = 224
model = ResNet50(weights='imagenet')
 

cap = cv2.VideoCapture(0)

# Configurar resolución alta
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Error: No se puede acceder a la cámara")
    exit()

# Capturar una imagen
ret, frame = cap.read()
if not ret:
    print("Error: No se pudo capturar la imagen")
    exit()

# Preprocesar la imagen para el modelo ResNet50
img = cv2.resize(frame, (224, 224))  # ResNet espera imágenes de 224x224
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
img = img_to_array(img)  # Convertir a un arreglo numpy
img = np.expand_dims(img, axis=0)  # Añadir una dimensión extra 
img = preprocess_input(img) 

# Hacer la predicción
preds = model.predict(img)
label = decode_predictions(preds)
print("Predicción", label[0][0][1])    

cv2.putText(frame, "{}, {:.1f}".format(label[0][0][1], label[0][0][2]) , (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3) 
# Mostrar la imagen capturada
cv2.imshow('Imagen Capturada', frame)
cv2.imwrite("captura.jpg", frame)

# Convertir a escala de grises
img = Image.open("captura.jpg").convert('L')
img.save('plot_grayscale.jpg', dpi=(300,300))
    
key = cv2.waitKey(0)
if key == 27:
   exit() 

# Liberar la cámara y cerrar ventanas
cap.release()
#cv2.destroyAllWindows()