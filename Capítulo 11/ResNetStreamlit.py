# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 16:17:43 2024

@author: Carlos Pineda
"""

import streamlit as st
import cv2 
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
 
from keras.applications.resnet50 import ResNet50, preprocess_input


def predecir(frame, tam_img):
    frame= cv2.resize(frame, (tam_img, tam_img))
    img_numpy = img_to_array(frame)
    img_exp = np.expand_dims(img_numpy, axis=0)
    img_procesada = preprocess_input(img_exp.copy())
 
    predicciones = modelo.predict(img_procesada)
    etiqueta_vgg = decode_predictions(predicciones)
    cv2.putText(frame, "{}, {:.2f}".format(etiqueta_vgg[0][0][1], etiqueta_vgg[0][0][2]) , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
    return frame


modelo = ResNet50(weights='imagenet')
tam_img = 224
 

st.title("Clasificador de Imagen")
st.sidebar.markdown("# Clasificación de imagen")
 
arc_img = st.sidebar.file_uploader("Cargue de imagen", type=['jpeg','jpg','png','gif'])
if arc_img is None:
    st.write("No hay archivo seleccionado!")
else:
    img = Image.open(arc_img)
    img = np.asarray(img)[:,:,::-1].copy() 
     
    img = predecir(img, tam_img)
    img = img[:,:,::-1]
    st.image(img, width=400)   
   