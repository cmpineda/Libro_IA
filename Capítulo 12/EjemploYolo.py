# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 04:31:15 2024

@author: Carlos Pineda
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

ruta_etiquetas = "coco.names"
ruta_pesos = "yolov3.weights"
ruta_config = "yolov3.cfg"
etiquetas = open(ruta_etiquetas).read().strip().split("\n")

np.random.seed(42)
colores = np.random.randint(0, 255, size=(len(etiquetas), 3), dtype="uint8")

escala = 0.005
umbral_confianza = 0.5
umbral_nms = 0.005  # Umbral de supresión no máxima
modelo = cv2.dnn.readNetFromDarknet(ruta_config, ruta_pesos)

imagen = cv2.imread("oficina.jpeg")
(alto_img, ancho_img) = imagen.shape[:2]

nom_capas = modelo.getLayerNames()
# capas de salida no conectadas
print(modelo.getUnconnectedOutLayers())

try:
    capas_salida = modelo.getUnconnectedOutLayers().flatten()  # Asegurarse de tener una lista plana
    nom_capas = [nom_capas[i - 1] for i in capas_salida]
except AttributeError:
    # Si `getUnconnectedOutLayers()` ya devuelve una lista, no es necesario hacer `flatten()`
    nom_capas = [nom_capas[i[0] - 1] for i in modelo.getUnconnectedOutLayers()]

#Redimensionamos la imagen a 416x416 pixels para poder pasar a través de la red
blob = cv2.dnn.blobFromImage(imagen, 1 / 255.0, (416, 416), swapRB=True, crop=False)
modelo.setInput(blob)

capas_finales = modelo.forward(nom_capas)

cajas = []
valores_confianzas = []
ids_clases = []

for salida in capas_finales:    
    for deteccion in salida:
        puntajes = deteccion[5:]
        id_clase = np.argmax(puntajes)
        confianza = puntajes[id_clase]

        if confianza > 0.15:
            #escalamos el tamaño del cuadro delimitador teniendo en cuenta
            #las coordenadas de la imagen
            caja = deteccion[0:4] * np.array([ancho_img, alto_img, ancho_img, alto_img])
            (centro_X, centro_Y, ancho, alto) = caja.astype("int")

            x = int(centro_X - (ancho / 2))
            y = int(centro_Y - (alto / 2))
            
            cajas.append([x, y, int(ancho), int(alto)])
            valores_confianzas.append(float(confianza))
            ids_clases.append(id_clase)
            
# aplicamos supresión no máxima
idxs = cv2.dnn.NMSBoxes(cajas, valores_confianzas, 0.6,0.2)

if len(idxs) > 0:
    
    for i in idxs.flatten():
        # extraemos las coordenadas de las cajas
        (x, y) = (cajas[i][0], cajas[i][1])
        (w, h) = (cajas[i][2], cajas[i][3])

        # Dibujamos el rectangulo y la etiqueta
        color = [int(c) for c in colores[ids_clases[i]]]
        cv2.rectangle(imagen, (x, y), (x + w, y + h), color, 6)
        texto = "{}: {:.4f}".format(etiquetas[ids_clases[i]], valores_confianzas[i])
        cv2.putText(imagen, texto, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            3, color, 6)
        
        
fig = plt.figure(figsize = (18,14))
plt.imshow(imagen[:,:,::-1])
plt.axis('off')
plt.savefig('oficina_yolo.jpg', dpi = 300)