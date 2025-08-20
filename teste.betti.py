import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.optimize import linear_sum_assignment

# ------------------------------
# PASTAS
# ------------------------------
pasta_curvas = r"C:\Users\crisl\Resul_ic\Curvas de Betti\csvs"
pasta_saida = r"C:\Users\crisl\Resultados_ic\KMeans_Betti"

os.makedirs(pasta_saida, exist_ok=True)

# ------------------------------
# CARREGAR CURVAS DE BETTI
# ------------------------------
arquivos = [f for f in os.listdir(pasta_curvas) if f.endswith('.csv')]
arquivos.sort()

dados = []
arquivos_validos = []

for nome_arquivo in arquivos:
    caminho = os.path.join(pasta_curvas, nome_arquivo)
    df = pd.read_csv(caminho)
    print("Lendo:", nome_arquivo, "| colunas:", df.columns)

    if "betti_0" not in df.columns or "betti_1" not in df.columns:
        print("⚠️ Pulando arquivo (colunas não encontradas):", nome_arquivo)
        continue

    vetor = df[["betti_0", "betti_1"]].values.flatten()

    if vetor.size == 0:
        print("⚠️ Vetor vazio em:", nome_arquivo)
        continue

    dados.append(vetor)
    arquivos_validos.append(nome_arquivo)

print("Total de curvas carregadas:", len(dados))


# Empilha em matriz (n_amostras x n_features)
X = np.vstack(dados)

# ------------------------------
# RÓTULOS VERDADEIROS
# ------------------------------
y_true = []
for nome in arquivos_validos:
    nome_lower = nome.lower()
    if "a_betti" in nome_lower:
        y_true.append(0)  # neutra
    elif "b_betti" in nome_lower:
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
# AJUSTAR CLUSTERS (Hungarian)
# ------------------------------
cm = confusion_matrix(y_true, y_pred)
row_ind, col_ind = linear_sum_assignment(-cm)
mapping = dict(zip(col_ind, row_ind))

y_pred_alinhado = np.array([mapping[c] for c in y_pred])

# ------------------------------
# MATRIZ DE CONFUSÃO
# ------------------------------
cm_final = confusion_matrix(y_true, y_pred_alinhado, labels=[0, 1])

disp = ConfusionMatrixDisplay(confusion_matrix=cm_final,
                              display_labels=["Neutra (a)", "Sorrindo (b)"])
disp.plot(cmap="Blues")
plt.title("Matriz de Confusão - Curvas de Betti")
plt.savefig(os.path.join(pasta_saida, "matriz_confusao_betti.png"))
plt.show()

# ------------------------------
# SALVAR RESULTADOS
# ------------------------------
df_resultados = pd.DataFrame({
    "arquivo": arquivos_validos,
    "classe_real": ["Neutra" if y == 0 else "Sorrindo" for y in y_true],
    "classe_predita": ["Neutra" if y == 0 else "Sorrindo" for y in y_pred_alinhado]
})
df_resultados.to_csv(os.path.join(pasta_saida, "resultados_classificacao_betti.csv"), index=False)
