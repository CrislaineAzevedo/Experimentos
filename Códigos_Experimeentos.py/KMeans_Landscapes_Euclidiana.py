import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from collections import defaultdict

pasta_landscapes = r"C:\Users\crisl\PIBIC\Landscape"
pasta_saida = r"C:\Users\crisl\PIBIC\KMeans_Landscapes"

os.makedirs(pasta_saida, exist_ok=True)

arquivos = [f for f in os.listdir(pasta_landscapes) if f.endswith('.npy')]
arquivos.sort()

dados = []
arquivos_validos = []

for nome_arquivo in arquivos:
    caminho = os.path.join(pasta_landscapes, nome_arquivo)
    vetor = np.load(caminho)

    if vetor.shape[0] != 8000:
        continue

    dados.append(vetor)
    arquivos_validos.append(nome_arquivo)

X = np.array(dados)
scores = []
ks = range(2, 26)

for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    rotulos = kmeans.fit_predict(X)
    score = silhouette_score(X, rotulos)
    scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(ks, scores, marker='o')
plt.title("Curva da Silhouette (Landscapes)")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Score da Silhouette")
plt.grid(True)
plt.savefig(os.path.join(pasta_saida, "curva_silhouette_landscapes.png"))
plt.close()

melhor_k = ks[np.argmax(scores)]

kmeans_final = KMeans(n_clusters=melhor_k, random_state=0, n_init='auto')
rotulos_finais = kmeans_final.fit_predict(X)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=rotulos_finais, cmap='tab20', s=60)
plt.title(f"Clusters das Landscapes (k = {melhor_k})")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True)
plt.savefig(os.path.join(pasta_saida, "clusters_PCA_landscapes.png"))
plt.close()

landscapes_por_cluster = defaultdict(list)

for vetor, r in zip(X, rotulos_finais):
    landscapes_por_cluster[r].append(vetor)

medias_landscapes = {}

for cluster_id in range(melhor_k):
    curvas = np.array(landscapes_por_cluster[cluster_id])
    medias_landscapes[cluster_id] = np.mean(curvas, axis=0)

plt.figure(figsize=(10, 6))
for cluster_id, curva in medias_landscapes.items():
    plt.plot(curva, label=f'Cluster {cluster_id}')
plt.title("Curvas médias das Landscapes por Cluster")
plt.xlabel("Índice da paisagem (landscape)")
plt.ylabel("Valor")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(pasta_saida, "curvas_medias_landscapes.png"))
plt.close()

pasta_csvs = os.path.join(pasta_saida, "curvas_medias_landscapes_csv")
os.makedirs(pasta_csvs, exist_ok=True)

for cluster_id, curva in medias_landscapes.items():
    df_saida = pd.DataFrame({'landscape': curva})
    nome_arquivo = f'landscape_media_cluster_{cluster_id}.csv'
    df_saida.to_csv(os.path.join(pasta_csvs, nome_arquivo), index=False)

df_rotulos = pd.DataFrame({
    'arquivo': arquivos_validos,
    'cluster': rotulos_finais
})
df_rotulos.to_csv(os.path.join(pasta_saida, "rotulos_clusters_landscapes.csv"), index=False)
