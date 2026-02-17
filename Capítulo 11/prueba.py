import cv2, os
from PIL import Image

from PIL import Image
# Convertir a escala de grises
img = Image.open("pantalla.jpg").convert('L')
img.save('gris.jpg', dpi=(300,300))