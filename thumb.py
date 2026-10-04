import os
from PIL import Image


def criar_mural_3x4_16_9(caminho_pasta, caminho_saida="mural_3x4_resultado.jpg"):
    extensoes_validas = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    # Lista e ordena os arquivos
    todos_arquivos = sorted(os.listdir(caminho_pasta))
    fotos = [
        arq
        for arq in todos_arquivos
        if arq.lower().endswith(extensoes_validas)
    ]

    total_fotos = len(fotos)
    print(f"Total de fotos encontradas: {total_fotos}")

    if total_fotos == 0:
        print("Nenhuma imagem encontrada.")
        return

    # Seleciona as 200 fotos com espaçamento igual
    if total_fotos < 200:
        fotos_selecionadas = fotos
    else:
        fotos_selecionadas = []
        for i in range(200):
            indice = int(i * (total_fotos - 1) / 99)
            fotos_selecionadas.append(fotos[indice])

    # --- CÁLCULO DA PROPORÇÃO 16:9 CRAVADO ---
    colunas = 13
    linhas = 8

    # Para o mural inteiro ter 16:9, definimos uma largura total (ex: 2080px)
    # 2080 / 13 colunas = 160px por miniatura
    # Para fechar 16:9, a altura total deve ser 1170px (2080 * 9 / 16)
    # 1170 / 8 linhas = ~146px por miniatura
    largura_mini = 160
    altura_mini = 146  # Ajustado para que o painel TOTAL seja 16:9

    largura_mural = colunas * largura_mini
    altura_mural = linhas * altura_mini

    # Cria o canvas do mural
    mural = Image.new("RGB", (largura_mural, altura_mural), color=(0, 0, 0))

    for idx, nome_foto in enumerate(fotos_selecionadas):
        caminho_foto = os.path.join(caminho_pasta, nome_foto)

        try:
            with Image.open(caminho_foto) as img:
                img = img.convert("RGB")

                # --- REDIMENSIONAMENTO COM CORTE CENTRAL (CROP) ---
                # Isso impede que a foto 3x4 fique distorcida ou esticada
                largura_orig, altura_orig = img.size
                proporcao_foco = largura_mini / altura_mini
                proporcao_orig = largura_orig / altura_orig

                if proporcao_orig > proporcao_foco:
                    # A foto original é mais larga do que precisamos
                    nova_largura = int(proporcao_orig * altura_mini)
                    img_redimensionada = img.resize(
                        (nova_largura, altura_mini), Image.Resampling.LANCZOS
                    )
                    # Corta as laterais
                    margin = (nova_largura - largura_mini) // 2
                    img_cortada = img_redimensionada.crop(
                        (margin, 0, margin + largura_mini, altura_mini)
                    )
                else:
                    # A foto original é mais alta (como as 3x4) do que precisamos
                    nova_altura = int(largura_mini / proporcao_orig)
                    img_redimensionada = img.resize(
                        (largura_mini, nova_altura), Image.Resampling.LANCZOS
                    )
                    # Corta o topo e o fundo (foco no centro da foto)
                    margin = (nova_altura - altura_mini) // 2
                    img_cortada = img_redimensionada.crop(
                        (0, margin, largura_mini, margin + altura_mini)
                    )

                # Calcula a posição na grade
                col = idx % colunas
                lin = idx // colunas

                x = col * largura_mini
                y = lin * altura_mini

                # Cola no mural
                mural.paste(img_cortada, (x, y))

        except Exception as e:
            print(f"Erro ao processar {nome_foto}: {e}")

    # Salva o resultado final
    mural.save(caminho_saida)
    print(
        f"\n🎉 Mural 16:9 gerado com sucesso! Resolução: {largura_mural}x{altura_mural}px"
    )
    print(f"Proporção exata alcançada: {largura_mural/altura_mural:.2f} (16:9 é 1.78)")


# --- CONFIGURAÇÃO ---
pasta_das_fotos = "./selfies"
criar_mural_3x4_16_9(pasta_das_fotos)