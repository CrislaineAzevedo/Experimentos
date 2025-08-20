import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


pasta_curvas = r"C:\Users\crisl\Resul_ic\Curvas de Betti\csvs"
arquivo_rotulos = r"C:\Users\crisl\Resul_ic\KMeans_Betti\rotulos_clusters.csv"
pasta_saida = r"C:\Users\crisl\Resultados_ic\Analise_Betti"


arquivos = [f for f in os.listdir(pasta_curvas) if f.endswith('.csv')]
arquivos.sort()

y_true = []
for nome in arquivos:
    nome_lower = nome.lower()
    if "a_betti" in nome_lower:
        y_true.append(0)  # Neutra
    elif "b_betti" in nome_lower:
        y_true.append(1)  # Sorrindo


df_pred = pd.read_csv(arquivo_rotulos)
df_pred = df_pred.set_index("arquivo").loc[arquivos]
y_pred = df_pred["cluster"].values


cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Neutra (a)", "Sorrindo (b)"])
disp.plot(cmap="Blues", values_format = "d")

plt.title("Matriz de Confusão - Curvas de Betti")
plt.ylabel("Rótulo real")
plt.xlabel("Rótulo previsto")
plt.tight_layout()
plt.savefig(os.path.join(pasta_saida, "matriz_confusao_betti.png"))
plt.show()
