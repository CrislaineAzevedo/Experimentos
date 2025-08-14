import numpy as np
import gudhi
import os
import pandas as pd
from tqdm import tqdm

# Caminhos
pasta_diagramas = r"C:\Users\crisl\PIBIC\Matriz_Diagrama"
pasta_saida = r"C:\Users\crisl\PIBIC\Distância_do_Gargalo"

# Garantir que a pasta de saída existe
os.makedirs(pasta_saida, exist_ok=True)

# Parâmetros
dimensoes = [0, 1]    # H0 e H1
k_max = 300
epsilon = 0.00001

def carregar_diagrama(caminho_arquivo, somente_dim=None):
    dados = np.load(caminho_arquivo)
    if somente_dim is not None:
        dados = dados[dados[:, 0] == somente_dim]
    return [(linha[1], linha[2]) for linha in dados]

def filtrar_mais_persistentes(diagrama, k):
    persistencias = [(b, d, d - b) for (b, d) in diagrama if d < float('inf')]
    persistencias.sort(key=lambda x: x[2], reverse=True)
    return [(b, d) for (b, d, _) in persistencias[:k]]

def distancia(dgm1, dgm2, e_aprox=None):
    try:
        if e_aprox is not None:
            return gudhi.bottleneck_distance(dgm1, dgm2, e_aprox)
        else:
            return gudhi.bottleneck_distance(dgm1, dgm2)
    except:
        return np.nan

# Lista de arquivos
arquivos = sorted([f for f in os.listdir(pasta_diagramas) if f.endswith(".npy")])
nomes = [os.path.splitext(f)[0] for f in arquivos]
n = len(arquivos)

# Loop para cada dimensão (H0, H1)
for dim in dimensoes:
    matriz = np.zeros((n, n))

    for i in tqdm(range(n), desc=f"Calculando distâncias H{dim}"):
        caminho1 = os.path.join(pasta_diagramas, arquivos[i])
        dgm1 = carregar_diagrama(caminho1, somente_dim=dim)
        dgm1 = filtrar_mais_persistentes(dgm1, k=k_max)

        for j in range(i, n):
            caminho2 = os.path.join(pasta_diagramas, arquivos[j])
            dgm2 = carregar_diagrama(caminho2, somente_dim=dim)
            dgm2 = filtrar_mais_persistentes(dgm2, k=k_max)

            if len(dgm1) == 0 or len(dgm2) == 0:
                dist = np.nan
            else:
                dist = distancia(dgm1, dgm2, e_aprox=epsilon)

            matriz[i, j] = dist
            matriz[j, i] = dist

    # Salvar CSV
    df = pd.DataFrame(matriz, index=nomes, columns=nomes)
    caminho_saida = os.path.join(pasta_saida, f"Distância_Gargalo_H{dim}.csv")
    df.to_csv(caminho_saida)
