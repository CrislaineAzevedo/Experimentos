from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

def dist_mahalanobis(pontos, p1, p2):
    covar = np.cov(pontos.T)
    inv_covar = np.linalg.pinv(covar)  
    dif = p1 - p2
    return np.sqrt(dif.T @ inv_covar @ dif)

def dist_euclidiana(p1, p2):
    return np.linalg.norm(p1 - p2)

def calcula_matriz_dist(pontos, metrica='Mahalanobis'):
    n = len(pontos)
    dist_matriz = np.zeros((n, n))
    
    if metrica == 'Mahalanobis':
        for i in range(n):
            for j in range(n):
                dist_matriz[i, j] = dist_mahalanobis(pontos, pontos[i], pontos[j])
    else:  
        for i in range(n):
            for j in range(n):
                dist_matriz[i, j] = dist_euclidiana(pontos[i], pontos[j])
    
    return dist_matriz

def vietoris_rips_complex(pontos, r, metrica = "Mahalanobis"):
    n = len(pontos)
    dist_matriz = calcula_matriz_dist(pontos, metrica)

    vertices = pontos  
    arestas = []
    triangulos = []

    for i, j in combinations(range(n), 2): 
        if dist_matriz[i, j] <= 2*r:
            arestas.append((pontos[i], pontos[j]))  

    for i, j, k in combinations(range(n), 3): 
        if (
            dist_matriz[i, j] <= 2*r and
            dist_matriz[j, k] <= 2*r and
            dist_matriz[i, k] <= 2*r
        ):
            triangulos.append((pontos[i], pontos[j], pontos[k]))  

    return vertices, arestas, triangulos

def plot_vietoris_rips_lado_a_lado(pontos, raio):
    metricas = ['Mahalanobis', 'Euclidiana']
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    for idx, metrica in enumerate(metricas):
        vertices, arestas, triangulos = vietoris_rips_complex(pontos, raio, metrica)

        axs[idx].scatter(vertices[:, 0], vertices[:, 1], color='black', zorder=3)
        
        for p1, p2 in arestas:
            axs[idx].plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', zorder=2)

        for p1, p2, p3 in triangulos:
            axs[idx].fill([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], color='blue', alpha=0.7, zorder=1)

        axs[idx].set_title(f"Métrica {metrica}", fontsize=14)
        axs[idx].grid(True)
        axs[idx].axis('equal')

    plt.tight_layout()
    plt.show()

(train_X, train_y), _ = mnist.load_data()
img = train_X[np.where(train_y == 8)[0][0]]

limiar = 100
coords = np.column_stack(np.where(img > limiar))
pontos = np.column_stack([coords[:,1], -coords[:,0]])
pontos = StandardScaler().fit_transform(pontos) 

# Mostrar a imagem original do dígito escolhido
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()


raio = 0.3
plot_vietoris_rips_lado_a_lado(pontos, raio)
