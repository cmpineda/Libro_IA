# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 21:43:16 2024

@author: Carlos Pineda
"""

import numpy as np
import cv2

ruta_etiquetas = "coco.names"
ruta_pesos = "yolov3.weights"
ruta_config = "yolov3.cfg"

# Cargar modelo YOLO
net = cv2.dnn.readNet(ruta_pesos, ruta_config)

# Cargar clases
etiquetas = open(ruta_etiquetas).read().strip().split("\n")

    # Abrir video
cap = cv2.VideoCapture("autos.mp4")  # 0 para webcam, "video.mp4" para archivo

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocesar frame
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

    # Establecer entrada del modelo
    net.setInput(blob)

    # Obtener salida del modelo
    salidas = net.forward(net.getUnconnectedOutLayersNames())

    # Detección de objetos
    ids_clases = []
    confianzas = []
    cajas = []
    for salida in salidas:
        for deteccion in salida:
            puntajes = deteccion[5:]
            id_clase = np.argmax(puntajes)
            confianza = puntajes[id_clase]
            if confianza > 0.5:  # Umbral de confianza y clase (en este caso, personas)
                x_centro = int(deteccion[0] * frame.shape[1])
                y_centro = int(deteccion[1] * frame.shape[0])
                w = int(deteccion[2] * frame.shape[1])
                h = int(deteccion[3] * frame.shape[0])
                x = int(x_centro - w / 2)
                y = int(y_centro - h / 2)
                cajas.append([x, y, w, h])
                confianzas.append(float(confianza))               
                ids_clases.append(id_clase)

    # Aplicar no-maxima supresión
    indices = cv2.dnn.NMSBoxes(cajas, confianzas, 0.5, 0.4)
    
    # Dibujar rectángulos alrededor de los objetos detectados
    for i in indices:        
        caja = cajas[i]
        x, y, w, h = caja[0], caja[1], caja[2], caja[3]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"{etiquetas[ids_clases[i]]} {confianzas[i]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Mostrar frame con objetos detectados
    cv2.imshow("Video", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()