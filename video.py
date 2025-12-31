import cv2
import os
from datetime import datetime, timedelta

# Configurações
image_folder = 'selfies_alinhadas'
output_video = 'video_com_datas.mp4'
fps = 15
initial_date = '2025-01-01'
date_format = '%b %d %Y'     # Formato: Jun 11 2020
date_color = (255, 255, 255) # ⚪ branco
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 8
thickness = 15
margin_top = 250

# Pega e ordena as imagens
images = sorted(
    [img for img in os.listdir(image_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
)

if len(images) == 0:
    raise ValueError(f"Nenhuma imagem encontrada em {image_folder}")

# Inicializa a data
current_date = datetime.strptime(initial_date, '%Y-%m-%d')

# Obtém o tamanho da imagem
first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = first_image.shape

# Inicializa o VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

n_files = len(os.listdir(image_folder))
i = 0

# Itera sobre as imagens
for img_name in images:
    i += 1
    print(f'{i}/{len(images)}')
    
    # if i == 6*31:
    #     break

    img_path = os.path.join(image_folder, img_name)
    img = cv2.imread(img_path)

    # ⚠️ Verifica leitura
    if img is None:
        print(f"⚠️ Erro ao ler {img_name}, pulando frame")
        current_date += timedelta(days=1)
        continue

    # ⚠️ Garante BGR
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # ⚠️ Remove canal alpha se existir
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # ⚠️ Garante tamanho correto
    if img.shape[0] != height or img.shape[1] != width:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    # Texto da data
    date_text = current_date.strftime(date_format)
    text_size = cv2.getTextSize(date_text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = margin_top

    cv2.putText(
        img,
        date_text,
        (text_x, text_y),
        font,
        font_scale,
        date_color,
        thickness,
        cv2.LINE_AA
    )

    video.write(img)
    current_date += timedelta(days=1)

# Finaliza o vídeo
video.release()
print(f"✅ Vídeo gerado com sucesso: {output_video}")
