import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from collections import defaultdict

pasta_curvas_betti = r"C:\Users\crisl\PIBIC\Curvas_de_Betti\csvs"
pasta_saida = r"C:\Users\crisl\PIBIC\KMeans_Betti"

arquivos = [f for f in os.listdir(pasta_curvas_betti) if f.endswith('.csv')]
arquivos.sort()

dados = []
raios_ref = None
arquivos_validos = []

for nome_arquivo in arquivos:
    caminho = os.path.join(pasta_curvas_betti, nome_arquivo)
    df = pd.read_csv(caminho)

    curva_raio = df['raio'].values
    curva_betti0 = df['betti_0'].values
    curva_betti1 = df['betti_1'].values

    if raios_ref is None:
        raios_ref = curva_raio
    elif not np.allclose(raios_ref, curva_raio, atol=1e-5):
        continue

    vetor = np.concatenate([curva_betti0, curva_betti1])
    dados.append(vetor)
    arquivos_validos.append(nome_arquivo)

X = np.array(dados)
n_pontos = len(raios_ref)

scores = []
ks = range(2, 26)

for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    rotulos = kmeans.fit_predict(X)
    score = silhouette_score(X, rotulos)
    scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(ks, scores, marker='o')
plt.title("Curva da Silhouette")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Score da Silhouette")
plt.grid(True)
plt.savefig(os.path.join(pasta_saida, "curva_silhouette.png"))
plt.close()

melhor_k = ks[np.argmax(scores)]

kmeans_final = KMeans(n_clusters=melhor_k, random_state=0, n_init='auto')
rotulos_finais = kmeans_final.fit_predict(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=rotulos_finais, cmap='tab20', s=60)
plt.title(f"Clusters das Curvas de Betti (k = {melhor_k})")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True)
plt.savefig(os.path.join(pasta_saida, "clusters_PCA.png"))
plt.close()

betti0_por_cluster = defaultdict(list)
betti1_por_cluster = defaultdict(list)

for vetor, r in zip(X, rotulos_finais):
    betti0 = vetor[:n_pontos]
    betti1 = vetor[n_pontos:]
    betti0_por_cluster[r].append(betti0)
    betti1_por_cluster[r].append(betti1)

medias_betti0 = {}
medias_betti1 = {}

for cluster_id in range(melhor_k):
    curvas0 = np.array(betti0_por_cluster[cluster_id])
    curvas1 = np.array(betti1_por_cluster[cluster_id])
    medias_betti0[cluster_id] = np.mean(curvas0, axis=0)
    medias_betti1[cluster_id] = np.mean(curvas1, axis=0)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for cluster_id, curva in medias_betti0.items():
    plt.plot(raios_ref, curva, label=f'Cluster {cluster_id}')
plt.title("Curvas médias - Betti 0")
plt.xlabel("Raio")
plt.ylabel("Número de componentes conexas")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for cluster_id, curva in medias_betti1.items():
    plt.plot(raios_ref, curva, label=f'Cluster {cluster_id}')
plt.title("Curvas médias - Betti 1")
plt.xlabel("Raio")
plt.ylabel("Número de buracos")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(pasta_saida, "curvas_medias_betti.png"))
plt.close()

pasta_csvs = os.path.join(pasta_saida, "curvas_medias_csv")
os.makedirs(pasta_csvs, exist_ok=True)

for cluster_id in range(melhor_k):
    df_saida = pd.DataFrame({
        'raio': raios_ref,
        'betti_0': medias_betti0[cluster_id],
        'betti_1': medias_betti1[cluster_id]
    })
    nome_arquivo = f'curva_media_cluster_{cluster_id}.csv'
    df_saida.to_csv(os.path.join(pasta_csvs, nome_arquivo), index=False)

df_rotulos = pd.DataFrame({
    'arquivo': arquivos_validos,
    'cluster': rotulos_finais
})
df_rotulos.to_csv(os.path.join(pasta_saida, "rotulos_clusters.csv"), index=False)


