import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2
import os


entrada = r"C:\Users\crisl\Rostos"
matriz_dist = r"C:\Users\crisl\Resultados_ic\Distância Euclidiana"
imagens = sorted(os.listdir(entrada))

limiar = 100
max_pontos = 5000

for nome_arquivo in tqdm(imagens, desc ="Calculando matrizes de distância"):
    caminho_img = os.path.join(entrada, nome_arquivo)
    img = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)

    coords = np.column_stack(np.where(img < limiar))
    pontos = np.column_stack([coords[:, 0], coords[:, 1]])
 
    if len(pontos) > max_pontos:
        indice = np.random.choice(len(pontos), max_pontos, replace=False)
        pontos = pontos[indice]

    #plt.figure(figsize = (6,4))
    #plt.scatter(pontos[:, 1], pontos[:, 0], s=1, c='blue')
    #plt.gca().invert_yaxis()
    #plt.show()

    N = len(pontos)
    dists = np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            dists[i][j] = np.linalg.norm(pontos[i] - pontos[j])

    nome_base = os.path.splitext(nome_arquivo)[0]
    caminho_saida = os.path.join(matriz_dist, f"{nome_base}.npy")
    np.save(caminho_saida, dists)