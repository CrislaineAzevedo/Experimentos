import numpy as np
import os
import matplotlib.pyplot as plt
from gudhi.representations import Landscape

entrada = r"C:\Users\crisl\Resultados_ic\Matriz_Diagrama"
saida = r"C:\Users\crisl\Resultados_ic\Landscapes"

num_landscape = 10 
pontos = 450   

todos_os_arquivos = os.listdir(entrada)

diagrama_arquivos = sorted(os.listdir(entrada))

for arquivo_nome in diagrama_arquivos:
    caminho_arquivo = os.path.join(entrada, arquivo_nome)
    
    try:
        dados_diagrama = np.load(caminho_arquivo)
        print(f'\nProcessando {arquivo_nome}: {dados_diagrama.shape[0]} pontos')
            
        # Extrair colunas de nascimento e morte (assumindo formato [dim, birth, death])
        diagrama_bd = dados_diagrama[:, 1:3].copy()
        
        # Tratar valores infinitos
        filtra_val_finitos = np.isfinite(diagrama_bd).all(axis=1)
        valores_finitos = diagrama_bd[filtra_val_finitos] #Seleciona apenas os pares nascimento e morte com valores finitos
        
        if valores_finitos.size > 0:
            max_finite = np.max(valores_finitos)
            diagrama_bd[~filtra_val_finitos] = max_finite * 2  # Substitui infinitos
        
        # Filtrar pontos válidos (death > birth)
        pontos_validos = (diagrama_bd[:, 1] > diagrama_bd[:, 0])
        diagrama_bd = diagrama_bd[pontos_validos]
            
        # Definir intervalo para as landscapes
        if diagrama_bd.shape[0] > 1:
            start = np.percentile(diagrama_bd[:, 0], 5) - 0.5  # Percentil 5
            stop = np.percentile(diagrama_bd[:, 1], 95) + 0.5   # Percentil 95
        else:
            start = diagrama_bd[0, 0] - 1
            stop = diagrama_bd[0, 1] + 1
        
        # Calcular Persistence Landscapes
        landscapes = Landscape(num_landscapes=num_landscape, resolution=pontos) #Quantas funções landscapes de nível k serão calculadas e quantos pontos iremos usar para discretizar o eixo x
        L = landscapes.fit_transform([diagrama_bd])
        xs = np.linspace(start, stop, pontos)
        
        plt.figure(figsize=(10, 5))
        
        for k in range(num_landscape):
            start_idx = k * pontos
            end_idx = (k + 1) * pontos
            plt.plot(xs, L[0, start_idx:end_idx], label=f'λ{k+1}')
        
        plt.title(f'Persistence Landscape: {os.path.splitext(arquivo_nome)[0]}')
        plt.xlabel('Filtração (ε)')
        plt.ylabel('λ(ε)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_file = os.path.join(saida, f"{os.path.splitext(arquivo_nome)[0]}_landscape.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        npy_saida = os.path.join(saida, f"{os.path.splitext(arquivo_nome)[0]}_landscape.npy")
        np.save(npy_saida, L[0])
        
    except Exception as e:
        print(f'  - Erro ao processar {arquivo_nome}: {str(e)}')

print('\nProcessamento concluído!')