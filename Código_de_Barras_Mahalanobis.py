import numpy as np
import matplotlib.pyplot as plt
import gudhi
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import inv
import os
import cv2
from tqdm import tqdm

entrada = r"C:\Users\crisl\Rostos"
saida = r"C:\Users\crisl\Resultados_ic\Código de Barras"
limiar = 100

def mahalanobis_distance_matrix(X):
    VI = inv(np.cov(X.T))  
    dists = pdist(X, metric='mahalanobis', VI=VI)
    return squareform(dists)

arquivos = os.listdir(entrada)

for nome_arquivo in tqdm(arquivos, desc="Processando imagens"):
    caminho_imagem = os.path.join(entrada, nome_arquivo)


    img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)

    coords = np.column_stack(np.where(img < limiar))
    pontos = np.column_stack([coords[:, 0], coords[:, 1]])

    if len(pontos) > 2000:
        idx = np.random.choice(len(pontos), 2000, replace=False)
        pontos = pontos[idx]

    # Complexo Rips com distância de Mahalanobis 
    dist_matrix = mahalanobis_distance_matrix(pontos)
    rips_maha = gudhi.RipsComplex(distance_matrix=dist_matrix, max_edge_length=1.0)
    st_maha = rips_maha.create_simplex_tree(max_dimension=2)
    diag_maha = st_maha.persistence()


    nome_saida = os.path.splitext(nome_arquivo)[0] + ".png"
    caminho_saida = os.path.join(saida, nome_saida)

    gudhi.plot_persistence_barcode(diag_maha)
    plt.title(f"Código de Barras - {os.path.splitext(nome_arquivo)[0]}")
    plt.savefig(caminho_saida)
    plt.close()


