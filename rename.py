import os


def padronizar_nomes_imagens(caminho_da_pasta):
    # Verifica se a pasta realmente existe
    if not os.path.exists(caminho_da_pasta):
        print(f"Erro: A pasta '{caminho_da_pasta}' não foi encontrada.")
        return

    # Lista todos os arquivos da pasta
    arquivos = os.listdir(caminho_da_pasta)

    contador_renomeados = 0

    for nome_arquivo in arquivos:
        # Caminho completo do arquivo atual
        caminho_antigo = os.path.join(caminho_da_pasta, nome_arquivo)

        # Garante que vai mexer apenas em arquivos (ignora subpastas)
        if os.path.isfile(caminho_antigo):

            # Verifica se o arquivo NÃO começa com 'IMG_'
            if not nome_arquivo.startswith("IMG_"):
                novo_nome = "IMG_" + nome_arquivo
                caminho_novo = os.path.join(caminho_da_pasta, novo_nome)

                # Renomeia o arquivo
                os.rename(caminho_antigo, caminho_novo)
                print(f"Renomeado: {nome_arquivo} -> {novo_nome}")
                contador_renomeados += 1

    print(
        f"\nProcesso concluído! {contador_renomeados} arquivos foram renomeados."
    )


# --- COMO USAR ---
# Substitua o caminho abaixo pelo caminho real da sua pasta de fotos
# Dica: No Windows, use 'r' antes das aspas se colar o caminho com barras invertidas (ex: r"C:\Usuarios\Fotos")
pasta_alvo = "./selfies"

padronizar_nomes_imagens(pasta_alvo)