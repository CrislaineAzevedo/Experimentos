import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.optimize import linear_sum_assignment

# ------------------------------
# PASTAS
# ------------------------------
pasta_landscapes = r"C:\Users\crisl\Resul_ic\Landscapes"
pasta_saida = r"C:\Users\crisl\Resultados_ic\KMeans_Landscapes"

os.makedirs(pasta_saida, exist_ok=True)

# ------------------------------
# CARREGAR LANDSCAPES
# ------------------------------
arquivos = [f for f in os.listdir(pasta_landscapes) if f.endswith('.npy')]
arquivos.sort()

dados = []
arquivos_validos = []

for nome_arquivo in arquivos:
    caminho = os.path.join(pasta_landscapes, nome_arquivo)
    vetor = np.load(caminho)

    # aceita qualquer vetor 1D
    if vetor.ndim != 1:
        continue

    dados.append(vetor)
    arquivos_validos.append(nome_arquivo)

X = np.array(dados)

# ------------------------------
# RÓTULOS VERDADEIROS
# ------------------------------
y_true = []
for nome in arquivos_validos:
    nome_lower = nome.lower()
    if "a_landscape" in nome_lower:
        y_true.append(0)  # neutra
    elif "b_landscape" in nome_lower:
        y_true.append(1)  # sorrindo
    else:
        raise ValueError(f"Nome inesperado: {nome}")

y_true = np.array(y_true)

# ------------------------------
# KMEANS COM 2 CLUSTERS
# ------------------------------
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
y_pred = kmeans.fit_predict(X)

# ------------------------------
# AJUSTAR CLUSTERS PARA CLASSES (Hungarian algorithm)
# ------------------------------
cm = confusion_matrix(y_true, y_pred)
row_ind, col_ind = linear_sum_assignment(-cm)
mapping = dict(zip(col_ind, row_ind))

# aplica mapeamento
y_pred_alinhado = np.array([mapping[c] for c in y_pred])

# ------------------------------
# MATRIZ DE CONFUSÃO
# ------------------------------
cm_final = confusion_matrix(y_true, y_pred_alinhado, labels=[0, 1])

disp = ConfusionMatrixDisplay(confusion_matrix=cm_final,
                              display_labels=["Neutra (a)", "Sorrindo (b)"])
disp.plot(cmap="Blues")
plt.title("Matriz de Confusão - Landscapes")
plt.savefig(os.path.join(pasta_saida, "matriz_confusao_landscapes.png"))
plt.show()

# ------------------------------
# SALVAR RESULTADOS
# ------------------------------
df_resultados = pd.DataFrame({
    "arquivo": arquivos_validos,
    "classe_real": ["Neutra" if y == 0 else "Sorrindo" for y in y_true],
    "classe_predita": ["Neutra" if y == 0 else "Sorrindo" for y in y_pred_alinhado]
})
df_resultados.to_csv(os.path.join(pasta_saida, "resultados_classificacao_landscapes.csv"), index=False)
