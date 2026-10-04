import cv2
import os
import numpy as np
from datetime import datetime, timedelta

# Configurações de Diretório e Datas
image_folder = 'selfies_alinhadas'
output_video = 'video_com_datas_youtube.mp4'
fps = 20
initial_date = '2020-06-11'
date_format = '%b %d %Y'     # Formato: Jun 11 2020
date_color = (255, 255, 255) # ⚪ branco

# Configurações do YouTube (Full HD 16:9)
yt_width = 1920
yt_height = 1080

# Ajustes da fonte (reduzidos pois a resolução final do vídeo é menor que a original)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
thickness = 5
margin_top = 100

# Pega e ordena as imagens
images = sorted(
    [img for img in os.listdir(image_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
)

if len(images) == 0:
    raise ValueError(f"Nenhuma imagem encontrada em {image_folder}")

# Inicializa a data
current_date = datetime.strptime(initial_date, '%Y-%m-%d')

# Descobrir a proporção da imagem original para redimensionar corretamente
first_image = cv2.imread(os.path.join(image_folder, images[0]))
orig_h, orig_w, _ = first_image.shape

# Calcula a nova largura para que a altura da foto encaixe perfeitamente nos 1080 pixels
aspect_ratio = orig_w / orig_h
new_w = int(yt_height * aspect_ratio)
new_h = yt_height

# Calcula a posição X para colar a foto bem no centro do fundo preto
x_offset = (yt_width - new_w) // 2

# Inicializa o VideoWriter com a resolução do YouTube
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, fps, (yt_width, yt_height))

n_files = len(images)

for i, img_name in enumerate(images, start=1):
    print(f'{i}/{n_files}')

    img_path = os.path.join(image_folder, img_name)
    img = cv2.imread(img_path)

    # ⚠️ Verifica leitura
    if img is None:
        print(f"⚠️ Erro ao ler {img_name}, pulando frame")
        current_date += timedelta(days=1)
        continue

    # ⚠️ Garante BGR e remove canal alpha se existir
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # 1. Redimensiona a selfie para caber na altura do YouTube
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 2. Cria o "Canvas" preto em Full HD
    canvas = np.zeros((yt_height, yt_width, 3), dtype=np.uint8)

    # 3. Cola a selfie redimensionada no centro do canvas preto
    canvas[0:yt_height, x_offset:x_offset+new_w] = img_resized

    # Texto da data (Calculado sobre a largura do canvas inteiro para ficar no meio da tela)
    date_text = current_date.strftime(date_format)
    text_size = cv2.getTextSize(date_text, font, font_scale, thickness)[0]
    text_x = (yt_width - text_size[0]) // 2
    text_y = margin_top

    cv2.putText(
        canvas,
        date_text,
        (text_x, text_y),
        font,
        font_scale,
        date_color,
        thickness,
        cv2.LINE_AA
    )

    # Grava o frame (que agora é a tela preta com a foto colada em cima)
    video.write(canvas)
    current_date += timedelta(days=1)

# Finaliza o vídeo
video.release()
print(f"✅ Vídeo gerado com sucesso: {output_video}")