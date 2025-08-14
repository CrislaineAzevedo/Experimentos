from tqdm import tqdm
import numpy as np
import gudhi
import cv2
import os

pasta_imagens = r"C:\Users\crisl\Rostos"
pasta_saidas = r"C:\Users\crisl\Resultados_ic\Matriz das Persistências"
limiar = 100

for nome_arquivo in tqdm(os.listdir(pasta_imagens)):
    caminho_imagem = os.path.join(pasta_imagens, nome_arquivo)

    img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
    coords = np.column_stack(np.where(img < limiar))

    max_pontos = 8000
    if coords.shape[0] > max_pontos:
        idx = np.random.choice(coords.shape[0], size=max_pontos, replace=False)
        coords = coords[idx]

    pontos = np.column_stack([coords[:, 0], coords[:, 1]])

    rips = gudhi.RipsComplex(points=pontos, max_edge_length=3.0)
    st = rips.create_simplex_tree(max_dimension=2)
    diag = st.persistence()

    # Converte para matriz: cada linha é (dim, birth, death)
    matriz = []
    for p in diag:
        dim, (birth, death) = p
        matriz.append([dim, birth, death])
    matriz = np.array(matriz)
   
    nome_base = os.path.splitext(nome_arquivo)[0]
    caminho_saida = os.path.join(pasta_saidas, nome_base + ".npy")
    np.save(caminho_saida, matriz)
