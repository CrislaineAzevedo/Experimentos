import numpy as np
import matplotlib.pyplot as plt
import gudhi
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import mnist
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import inv

(train_X, train_y), _ = mnist.load_data()
img = train_X[np.where(train_y == 8)[0][0]]

limiar = 100
coords = np.column_stack(np.where(img < limiar))
pontos = np.column_stack([coords[:, 1], -coords[:, 0]])
pontos = StandardScaler().fit_transform(pontos)


# Distância de Mahalanobis 
def mahalanobis_distance_matrix(X):
    VI = inv(np.cov(X.T)) 
    dists = pdist(X, metric='mahalanobis', VI=VI)
    return squareform(dists)

# Complexo Rips com distância Euclidiana 
rips_euclid = gudhi.RipsComplex(points=pontos, max_edge_length=1.0)
st_euclid = rips_euclid.create_simplex_tree(max_dimension=2)
diag_euclid = st_euclid.persistence()

# Complexo Rips com distância de Mahalanobis 
dist_matrix = mahalanobis_distance_matrix(pontos)
rips_maha = gudhi.RipsComplex(distance_matrix=dist_matrix, max_edge_length=1.0)
st_maha = rips_maha.create_simplex_tree(max_dimension=2)
diag_maha = st_maha.persistence()


fig, axs = plt.subplots(1, 2, figsize=(12, 5))

gudhi.plot_persistence_barcode(diag_euclid, axes=axs[0])
axs[0].set_title("Código de Barras - Euclidiana")

gudhi.plot_persistence_barcode(diag_maha, axes=axs[1])
axs[1].set_title("Código de Barras - Mahalanobis")

plt.tight_layout()
plt.show()



fig, axs = plt.subplots(1, 2, figsize=(12, 5))

gudhi.plot_persistence_diagram(diag_euclid, axes=axs[0])
axs[0].set_title("Diagrama de Persistência - Euclidiana")

gudhi.plot_persistence_diagram(diag_maha, axes=axs[1])
axs[1].set_title("Diagrama de Persistência - Mahalanobis")

plt.tight_layout()
plt.show()
