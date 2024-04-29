import numpy as np

# Metode dekomposisi LU Gauss
def metode_lu_gauss(A, b):
    n = len(A) 
    L = np.eye(n) 
    U = A.copy().astype(float)  

    # Perulangan eliminasi Gauss
    for k in range(n-1):
        for i in range(k+1, n):
            faktor = U[i, k] / U[k, k]
            L[i, k] = faktor
            U[i, k:] -= faktor * U[k, k:]
    
    # Perhitungan solusi
    y = np.linalg.solve(L, b.astype(float))  
    x = np.linalg.solve(U, y)
    return x

# Kode pengujian
A = np.array([[3, 2, -2],
              [-1, -2, 3],
              [2, 1, 2]])
b = np.array([6, -5, 4])

print("\nMetode Dekomposisi LU Gauss:")
print(metode_lu_gauss(A, b))
