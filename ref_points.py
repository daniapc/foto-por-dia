import cv2
import dlib
import numpy as np
import os
from imutils import face_utils

# Caminhos
input_folder = 'selfies'
output_folder = 'selfies_alinhadas'
predictor_path = 'shape_predictor_68_face_landmarks.dat'

# Inicializar detector e preditor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

os.makedirs(output_folder, exist_ok=True)

ref_points = None
count = 0

# 1. Calcular a média de todos os pontos faciais
for filename in sorted(os.listdir(input_folder)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image = cv2.imread(os.path.join(input_folder, filename))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        if len(rects) == 0:
            continue

        # Selecionar rosto mais central
        h, w = gray.shape
        center_x, center_y = w // 2, h // 2

        def dist(rect):
            x = (rect.left() + rect.right()) / 2
            y = (rect.top() + rect.bottom()) / 2
            return (x - center_x) ** 2 + (y - center_y) ** 2

        rect = min(rects, key=dist)

        shape = predictor(gray, rect)
        shape_np = face_utils.shape_to_np(shape)

        if ref_points is None:
            ref_points = shape_np.astype(np.float32)
        else:
            ref_points += shape_np.astype(np.float32)
        count += 1

ref_points /= count  # média de pontos
print(ref_points)

np.save('ref_points.npy', ref_points)