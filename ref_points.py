import os
import cv2
import numpy as np

# A importação da API de inferência do MMPose
from mmpose.apis import MMPoseInferencer

def main():
    print("Inicializando MMPose...")
    # O parâmetro pose2d='face' avisa o MMPose para baixar e usar o modelo RTMPose otimizado para rostos
    inferencer = MMPoseInferencer(pose2d='face')
    print("MMPose carregado com sucesso!")

    input_folder = 'selfies'
    ref_points = None
    count = 0
    target_size = (1944, 2592)

    if not os.path.exists(input_folder):
        print(f"Erro: A pasta '{input_folder}' não foi encontrada.")
        return

    arquivos = [f for f in sorted(os.listdir(input_folder)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    n_files = len(arquivos)
    
    if n_files == 0:
        print("Nenhuma imagem encontrada.")
        return

    for i, filename in enumerate(arquivos, start=1):
        print(f'Processando imagem {i}/{n_files}: {filename}')

        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Falha ao ler {filename}")
            continue

        image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

        # O MMPoseInferencer funciona como um gerador (yield). 
        # Chamamos next() para pegar o resultado da imagem atual.
        resultado_gen = inferencer(image_resized, show=False)
        resultado = next(resultado_gen)
        
        # Estrutura do resultado: dict com a chave 'predictions'
        # predictions[0] contém as instâncias (pessoas/rostos) achadas na imagem
        predicoes = resultado['predictions'][0]

        if len(predicoes) == 0:
            print(f"Nenhum rosto encontrado em {filename}")
            continue

        # Vamos pegar o primeiro rosto detectado (índice 0)
        # 'keypoints' é uma lista com as coordenadas [x, y]
        pontos = np.array(predicoes[0]['keypoints'], dtype=np.float32)

        if ref_points is None:
            ref_points = pontos
        else:
            ref_points += pontos

        count += 1

    if count > 0:
        ref_points /= count
        print("\n=== SUCESSO ===")
        print(f"Média calculada com base em {count} rostos.")
        print(f"O MMPose detectou {ref_points.shape[0]} pontos faciais em cada rosto.")
        
        np.save('ref_points.npy', ref_points)
        print("Arquivo 'ref_points.npy' salvo!")

if __name__ == '__main__':
    main()