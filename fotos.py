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

ref_points = np.load('ref_points.npy')

# 2. Alinhar cada imagem com base nos 68 pontos
for filename in sorted(os.listdir(input_folder)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 1)
        if len(rects) == 0:
            print(f"Nenhum rosto encontrado em {filename}")
            continue

        h, w = gray.shape
        center_x, center_y = w // 2, h // 2
        rect = min(
            rects,
            key=lambda r: ((r.left() + r.right()) / 2 - center_x) ** 2 + ((r.top() + r.bottom()) / 2 - center_y) ** 2
        )

        shape = predictor(gray, rect)
        shape_np = face_utils.shape_to_np(shape)

        # Calcular transformação de similaridade
        M, _ = cv2.estimateAffinePartial2D(shape_np.astype(np.float32), ref_points)


        # aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

        # Aplicar warp na imagem
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

        # Criar máscara da área válida da imagem transformada
        mask_gray = np.full(gray.shape, 255, dtype=np.uint8)  # imagem branca
        mask_warped = cv2.warpAffine(mask_gray, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)

        # Converter para 3 canais para fazer np.where
        mask_3ch = cv2.merge([mask_warped] * 3)

        # Se for o primeiro frame, salvar como anterior
        if 'previous_frame' not in locals():
            previous_frame = aligned.copy()

        # Combinar com a imagem anterior onde não há dados (mascara preta)
        combined = np.where(mask_3ch == 0, previous_frame, aligned)

        # Atualizar imagem anterior
        previous_frame = combined.copy()

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, aligned)