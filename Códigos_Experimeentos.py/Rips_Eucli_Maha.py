import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis, euclidean

# Calcula a distância de Mahalanobis entre dois pontos
def Dist_Mahalanobis(pontos, p1, p2):
    covar = np.cov(np.array([pontos[:, 0], pontos[:, 1]]))
    inv_covar = np.linalg.inv(covar)
    dif = np.array([p1 - p2])
    dif_t = dif.transpose()
    return np.sqrt(dif @ inv_covar @ dif_t)[0, 0]

# Calcula a distância Euclidiana entre dois pontos
def Dist_Euclidiana(p1, p2):
    return np.linalg.norm(p1 - p2)

# Calcula a matriz de distância
def calcula_matriz_dist(pontos, metrica='Mahalanobis'):
    n = len(pontos)
    dist_matriz = np.zeros((n, n))

    if metrica == 'Mahalanobis':
        for i in range(n):
            for j in range(n):
                dist_matriz[i, j] = Dist_Mahalanobis(pontos, pontos[i], pontos[j])
    else:
        for i in range(n):
            for j in range(n):
                dist_matriz[i, j] = Dist_Euclidiana(pontos[i], pontos[j])

    return dist_matriz

# Complexo de Vietoris-Rips
def vietoris_rips_complex(pontos, r, metrica="Mahalanobis"):
    n = len(pontos)
    dist_matriz = calcula_matriz_dist(pontos, metrica)

    vertices = pontos
    arestas = []
    triangulos = []

    for i, j in combinations(range(n), 2):
        if dist_matriz[i, j] <= 2 * r:
            arestas.append((pontos[i], pontos[j]))

    for i, j, k in combinations(range(n), 3):
        if (
            dist_matriz[i, j] <= 2 * r and
            dist_matriz[j, k] <= 2 * r and
            dist_matriz[i, k] <= 2 * r
        ):
            triangulos.append((pontos[i], pontos[j], pontos[k]))

    return vertices, arestas, triangulos

# Figura 8
def generate_figure_8(n_points, epsilon):
    t = np.linspace(0, 2 * np.pi, n_points // 2, endpoint=False)

    h = 1
    x1 = np.sin(t)
    y1 = np.cos(t) + h
    x2 = np.sin(t)
    y2 = -np.cos(t) - h

    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])

    x += np.random.uniform(-epsilon, epsilon, size=x.shape)
    y += np.random.uniform(-epsilon, epsilon, size=y.shape)

    return np.column_stack((x, y))

# Dados
n_pontos = 20
epsilon = 0.2
raio = 0.7
pontos = generate_figure_8(n_pontos, epsilon)

# Plot
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

for idx, metrica in enumerate(['Mahalanobis', 'Euclidiana']):
    vertices, arestas, triangulos = vietoris_rips_complex(pontos, raio, metrica)

    axs[idx].scatter(pontos[:, 0], pontos[:, 1], color='black', label='Pontos', zorder=3)
    for p1, p2 in arestas:
        axs[idx].plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', zorder=2)
    for p1, p2, p3 in triangulos:
        axs[idx].fill([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], color='blue', alpha=0.7, zorder=1)

    axs[idx].set_title(f"Métrica\n{metrica}", fontsize=12)
    axs[idx].grid(True)
    axs[idx].axis('equal')

plt.tight_layout()
plt.show()
