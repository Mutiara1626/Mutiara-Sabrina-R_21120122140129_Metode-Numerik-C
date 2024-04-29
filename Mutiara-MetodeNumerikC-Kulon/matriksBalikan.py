import numpy as np

# Metode matriks balikan
def metode_matriks_balikan(A, b):
    A_inv = np.linalg.inv(A) # untuk menghitung invers dari matriks koefisien yaitu A
    x = np.dot(A_inv, b) # mengalikan invers matriks A dengan vektor hasil yaitu b
    return x

# Kode testing
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])
b = np.array([8, -11, -3])

print("Metode Matriks Balikan:")
print(metode_matriks_balikan(A, b))