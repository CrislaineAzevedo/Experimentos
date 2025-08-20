import cv2
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import gudhi
from zipfile import ZipFile

pasta_rostos = r"C:\Users\crisl\Rostos"
pasta_diagrama = r"C:\Users\crisl\Resultados_ic\Diagrama de Persistência"
limiar = 100

for nome_arquivo in os.listdir(pasta_rostos):
    caminho_imagem = os.path.join(pasta_rostos, nome_arquivo)


    img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)


    coords = np.column_stack(np.where(img < limiar))
    pontos = np.column_stack([coords[:, 0], coords[:, 1]])
 
    if len(pontos) > 8000:
        idx = np.random.choice(len(pontos), 8000, replace=False)
        pontos = pontos[idx]

    rips_euclid = gudhi.RipsComplex(points=pontos, max_edge_length=3.0)
    st_euclid = rips_euclid.create_simplex_tree(max_dimension=2)
    diag_euclid = st_euclid.persistence()


    nome_saida = os.path.splitext(nome_arquivo)[0] + ".png"
    caminho_saida = os.path.join(pasta_diagrama, nome_saida)

    gudhi.plot_persistence_diagram(diag_euclid)
    plt.title(f"Diagrama de persistência - {os.path.splitext(nome_arquivo)[0]}")
    plt.savefig(caminho_saida)
    plt.close()

