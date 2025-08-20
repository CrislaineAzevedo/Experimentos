from tqdm import tqdm
import numpy as np
import pickle
import gudhi
import gc
import os

# Distância de Mahalanobis de cada imagem usando 2500 pontos
entrada = r"C:\Users\crisl\Resultados_ic\Distância de Mahalanobis"

saida = r"C:\Users\crisl\Resultados_ic\Persistências"

arquivos = [f for f in os.listdir(entrada) if f.endswith(".npy")]


for nome_arquivo in tqdm(arquivos, desc = "Salvando Persistências"):

    caminho = os.path.join(entrada, nome_arquivo)
    dist_matriz = np.load(caminho)

    rips_maha = gudhi.RipsComplex(distance_matrix=dist_matriz, max_edge_length=0.8)
    st_maha = rips_maha.create_simplex_tree(max_dimension=2)
    diag_maha = st_maha.persistence()

    nome_saida = os.path.splitext(nome_arquivo)[0] + ".pkl"
    caminho_saida = os.path.join(saida, nome_saida)

    with open(caminho_saida, "wb") as f:
        pickle.dump(diag_maha, f)

    del dist_matriz, rips_maha, st_maha, diag_maha
    gc.collect()



