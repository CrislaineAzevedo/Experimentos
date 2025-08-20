import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import gudhi
import os


def GetBettiCurvesFromDistances(D, J, dim=2):
    I = 2*J
    tmax = max(I)
    rips = gudhi.RipsComplex(distance_matrix = D, max_edge_length = tmax)
    st = rips.create_simplex_tree(max_dimension=dim)
    st.persistence(persistence_dim_max=True, homology_coeff_field = 2)
    Diagrams = [st.persistence_intervals_in_dimension(i) for i in range(dim+1)]
    BettiCurves = []
    step_x = I[1]-I[0]
    for diagram in Diagrams:
        bc =  np.zeros(len(I))
        if diagram.size != 0:
            diagram_int = np.clip(np.ceil((diagram[:,:2] - I[0]) / step_x), 0, len(I)).astype(int)
            for interval in diagram_int:
                bc[interval[0]:interval[1]] += 1
        BettiCurves.append(np.reshape(bc,[1,-1]))
    return np.reshape(BettiCurves, (dim+1, len(I)))



matriz_distancia = r"C:\Users\crisl\Resultados_ic\Distância Euclidiana"
saida_img = r"C:\Users\crisl\Resul_ic\Curvas de Betti\plots"
saida_csv = r"C:\Users\crisl\Resul_ic\Curvas de Betti\csvs"

raios = np.linspace(0.05, 3, 100) 
matrizes = sorted([f for f in os.listdir(matriz_distancia) if f.endswith(".npy")])

for nome_arq in tqdm(matrizes, desc="Calculando curvas de Betti"):
    caminho = os.path.join(matriz_distancia, nome_arq)
    D = np.load(caminho)

    curva = GetBettiCurvesFromDistances(D, raios, dim=2) 

    df = pd.DataFrame({
        "raio": raios,
        "betti_0": curva[0],
        "betti_1": curva[1]
    })

    nome_base = os.path.splitext(nome_arq)[0]
    df.to_csv(os.path.join(saida_csv, f"{nome_base}_betti.csv"), index=False)

    plt.figure()
    plt.plot(raios, curva[0], label=r"$\beta_0$")
    plt.plot(raios, curva[1], label=r"$\beta_1$")
    plt.title(f"Curva de Betti - {nome_base}")
    plt.xlabel("Raio $r$")
    plt.ylabel("Número de Betti")
    plt.legend()
    plt.tight_layout()

    caminho_plot = os.path.join(saida_img, f"{nome_base}.png")
    plt.savefig(caminho_plot)
    plt.close()

print(f"Curvas de Betti salvas em: {saida_img}")
print(f"Arquivos CSV salvos em: {saida_csv}")
