import cv2
import dlib
import numpy as np
import os
from imutils import face_utils

# Caminhos
input_folder = 'selfies'
output_folder = 'selfies_alinhadas'
predictor_path = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

os.makedirs(output_folder, exist_ok=True)

ref_points = np.load('ref_points.npy')

# Índices: contorno + olhos + ponta do nariz (mesmo que na geração)
contorno_indices = list(range(0, 17))
olho_esq_indices = list(range(36, 42))
olho_dir_indices = list(range(42, 48))
nariz_indices = [30]
selected_indices = contorno_indices + olho_esq_indices + olho_dir_indices + nariz_indices

target_width = 1944
target_height = 2592
target_size = (target_width, target_height)

n_files = len(os.listdir(input_folder))
i = 0

for filename in sorted(os.listdir(input_folder)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        i += 1
        print(f'{str(i)}/{str(n_files)}')

        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(image_resized, -1, kernel)

        gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        rects = detector(gray, 1)

        if len(rects) == 0:
            print(f"Nenhum rosto encontrado em {filename}")
            continue

        h, w = gray.shape
        center_x, center_y = w // 2, h // 2

        def score(rect):
            x = (rect.left() + rect.right()) / 2
            y = (rect.top() + rect.bottom()) / 2
            dist_to_center = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            area = (rect.right() - rect.left()) * (rect.bottom() - rect.top())
            return dist_to_center - 0.3 * area

        rect = min(rects, key=score)

        shape = predictor(gray, rect)
        shape_np = face_utils.shape_to_np(shape)

        selected_points = shape_np[selected_indices]

        M, _ = cv2.estimateAffinePartial2D(selected_points.astype(np.float32), ref_points)

        aligned = cv2.warpAffine(image_resized, M, target_size, flags=cv2.INTER_CUBIC)

        mask_gray = np.full(gray.shape, 255, dtype=np.uint8)
        mask_warped = cv2.warpAffine(mask_gray, M, target_size, flags=cv2.INTER_NEAREST)
        mask_3ch = cv2.merge([mask_warped] * 3)

        if 'previous_frame' not in locals():
            previous_frame = aligned.copy()

        combined = np.where(mask_3ch == 255, aligned, previous_frame)
        previous_frame = combined.copy()

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, combined)
