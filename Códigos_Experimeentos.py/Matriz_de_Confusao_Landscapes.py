import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.optimize import linear_sum_assignment

landscapes = r"C:\Users\crisl\Resul_ic\Landscapes"
rotulos = r"C:\Users\crisl\Resul_ic\KMeans_Landscapes\rotulos_clusters_landscapes.csv"
saida = r"C:\Users\crisl\Resultados_ic\Matriz de Confusão - Landscapes"

# ------------------------------
# Rótulos reais
# ------------------------------
arquivos = sorted([f for f in os.listdir(landscapes) if f.endswith('.npy')])
y_true = [0 if "a_landscape" in nome.lower() else 1 for nome in arquivos]

# ------------------------------
# Rótulos previstos
# ------------------------------
df_pred = pd.read_csv(rotulos).set_index("arquivo").loc[arquivos]
y_pred = df_pred["cluster"].values

# ------------------------------
# Ajustar clusters (Hungarian Algorithm)
# ------------------------------
cm = confusion_matrix(y_true, y_pred)
row_ind, col_ind = linear_sum_assignment(-cm)
mapping = dict(zip(col_ind, row_ind))

# aplica mapeamento
y_pred_alinhado = np.array([mapping[c] for c in y_pred])

# ------------------------------
# Matriz de confusão final (corrigida)
# ------------------------------
cm_final = confusion_matrix(y_true, y_pred_alinhado, labels=[0, 1])

disp = ConfusionMatrixDisplay(confusion_matrix=cm_final,
                              display_labels=["Neutra (a)", "Sorrindo (b)"])
disp.plot(cmap="Blues", values_format = "d")

plt.title("Matriz de Confusão - Persistence Landscapes")
plt.ylabel("Rótulo real")
plt.xlabel("Rótulo previsto")
plt.tight_layout()
plt.savefig(os.path.join(saida, "matriz_confusao_landscapes.png"))
plt.show()
