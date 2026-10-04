import os
import cv2
import numpy as np

# Trava para evitar Fork Bomb no Linux
os.environ["OMP_NUM_THREADS"] = "1"

from mmpose.apis import MMPoseInferencer

def main():
    print("Inicializando MMPose para alinhamento...")
    inferencer = MMPoseInferencer(pose2d='face')
    print("MMPose carregado com sucesso!")

    # Caminhos
    input_folder = 'fotos_2021'
    output_folder = 'selfies_alinhadas'
    os.makedirs(output_folder, exist_ok=True)

    # Carregar os pontos de referência salvos no script anterior
    try:
        ref_points = np.load('ref_points.npy')
    except FileNotFoundError:
        print("Erro: 'ref_points.npy' não encontrado. Rode o script de média primeiro!")
        return

    target_width = 1944
    target_height = 2592
    target_size = (target_width, target_height)

    arquivos = [f for f in sorted(os.listdir(input_folder)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    n_files = len(arquivos)
    
    if n_files == 0:
        print("Nenhuma imagem encontrada.")
        return

    previous_frame = None
    anchor_mean = None
    anchor_std = None

    for i, filename in enumerate(arquivos, start=1):
        print(f'Alinhando imagem {i}/{n_files}: {filename}')

        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Falha ao ler {filename}")
            continue

        # Redimensionar para o tamanho fixo 
        image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

        # Inferência com MMPose
        resultado_gen = inferencer(image_resized, show=False)
        resultado = next(resultado_gen)
        
        predicoes = resultado['predictions'][0]

        if len(predicoes) == 0:
            print(f"Nenhum rosto encontrado em {filename}. Usando frame anterior (se existir).")
            if previous_frame is not None:
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, previous_frame)
            continue

        # Pegar os pontos do primeiro rosto detectado
        pontos = np.array(predicoes[0]['keypoints'], dtype=np.float32)

        # Calcular a matriz de transformação Afim
        M, _ = cv2.estimateAffinePartial2D(pontos, ref_points)

        if M is None:
            print(f"Falha ao calcular transformação para {filename}")
            continue

        # Aplicar o alinhamento
        aligned = cv2.warpAffine(image_resized, M, target_size, flags=cv2.INTER_CUBIC)

        # --- AJUSTE LEVE DE LUMINOSIDADE E CONTRASTE ---
        aligned_lab = cv2.cvtColor(aligned, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(aligned_lab)
        
        if anchor_mean is None:
            # Na primeira foto, guardamos a Média (brilho) e o Desvio Padrão (contraste)
            anchor_mean = np.mean(l)
            anchor_std = np.std(l)
            aligned_adjusted = aligned.copy()
        else:
            l_mean = np.mean(l)
            l_std = np.std(l)
            
            # Evita divisão por zero
            if l_std == 0:
                l_std = 1e-6
            
            # Ajuste linear suave: estica/encolhe o contraste e desloca o brilho
            l_adjusted = (l - l_mean) * (anchor_std / l_std) + anchor_mean
            
            # Corta valores fora do limite 0-255 e volta para o formato de imagem
            l_adjusted = np.clip(l_adjusted, 0, 255).astype(np.uint8)
            
            # Recombina com as cores originais (a, b) que ficaram 100% intocadas
            lab_adjusted = cv2.merge((l_adjusted, a, b))
            aligned_adjusted = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)
        # -----------------------------------------------

        # Lógica de preenchimento de bordas usando a imagem com luz ajustada
        mask_gray = np.full((target_height, target_width), 255, dtype=np.uint8)
        mask_warped = cv2.warpAffine(mask_gray, M, target_size, flags=cv2.INTER_NEAREST)
        mask_3ch = cv2.merge([mask_warped] * 3)

        if previous_frame is None:
            previous_frame = aligned_adjusted.copy()

        combined = np.where(mask_3ch == 255, aligned_adjusted, previous_frame)
        
        # Atualiza o frame anterior para a próxima iteração
        previous_frame = combined.copy()

        # Salvar o resultado
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, combined)

    print("\n=== PROCESSO CONCLUÍDO ===")
    print(f"Imagens alinhadas salvas na pasta: {output_folder}")

if __name__ == '__main__':
    main()
