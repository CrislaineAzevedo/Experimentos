import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gudhi
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import inv
import os
import cv2  


imagem = r"C:\Users\crisl\Rostos"
matriz_dist = r"C:\Users\crisl\Resultados_ic\Distância de Mahalanobis"
arquivos_imagem = sorted(os.listdir(imagem))

limiar = 100
max_pontos = 2500

def mahalanobis_distance_matrix(X):
    VI = inv(np.cov(X.T))  
    dists = pdist(X, metric='mahalanobis', VI=VI)
    return squareform(dists)

for nome_arquivo in tqdm(arquivos_imagem, desc ="Calculando matrizes de distância"):

    caminho_imagem = os.path.join(imagem, nome_arquivo)
    img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
    coords = np.column_stack(np.where(img < limiar))

    pontos = StandardScaler().fit_transform(coords)

    if len(pontos) > max_pontos:
        indice = np.random.choice(len(pontos), max_pontos, replace=False)
        pontos = pontos[indice]

    #plt.figure(figsize = (6,4))
    #plt.scatter(pontos[:, 1], pontos[:, 0], s=1, c='blue')
    #plt.gca().invert_yaxis()
    #plt.show()
    
    dists = mahalanobis_distance_matrix(pontos)
    
    
    nome_base = os.path.splitext(nome_arquivo)[0]
    caminho_saida = os.path.join(matriz_dist, f"{nome_base}.npy")
    np.save(caminho_saida, dists)
    
print(f"✅ Matriz de distância salva: {caminho_saida}")
