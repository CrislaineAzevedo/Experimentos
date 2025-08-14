from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from zipfile import ZipFile
import numpy as np
import gudhi
import cv2
import os

entrada = r"C:\Users\crisl\Rostos"
saida = r"C:\Users\crisl\Resultados_ic\Código de Barras"
limiar = 100

for nome_arquivo in os.listdir(entrada):
    caminho_imagem = os.path.join(entrada, nome_arquivo)


    img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)

    coords = np.column_stack(np.where(img < limiar))
    pontos = np.column_stack([coords[:, 0], coords[:, 1]])

    np.random.seed(42) 

    if len(pontos) > 8000:
        idx = np.random.choice(len(pontos), 8000, replace=False)
        pontos = pontos[idx]

    rips_euclid = gudhi.RipsComplex(points=pontos, max_edge_length=3.0)
    st_euclid = rips_euclid.create_simplex_tree(max_dimension=2)
    diag_euclid = st_euclid.persistence()


    nome_saida = os.path.splitext(nome_arquivo)[0] + ".png"
    caminho_saida = os.path.join(saida, nome_saida)

    gudhi.plot_persistence_barcode(diag_euclid)
    plt.title(f"Código de Barras - {os.path.splitext(nome_arquivo)[0]}")
    plt.savefig(caminho_saida)
    plt.close()


