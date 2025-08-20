import numpy as np
betti = np.array([[53, 47],
                  [30, 70]])

landscapes = np.array([[65, 35],
                       [25, 75]])

def accuracy(matrix):
    return (matrix[0,0] + matrix[1,1]) / matrix.sum()

acc_betti = accuracy(betti)
acc_landscapes = accuracy(landscapes)

diff = acc_landscapes - acc_betti

print("Acurácia Betti:", acc_betti)
print("Acurácia Landscapes:", acc_landscapes)
print("Diferença:", diff)
