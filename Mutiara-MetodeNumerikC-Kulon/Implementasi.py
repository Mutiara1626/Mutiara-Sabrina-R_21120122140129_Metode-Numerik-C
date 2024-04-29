import numpy as np

# Metode matriks balikan
def metode_matriks_balikan(A, b):
    A_inv = np.linalg.inv(A) 
    x = np.dot(A_inv, b) 
    return x

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
    y = np.linalg.solve(L, b.astype(float))  # Mengonversi b ke tipe float
    x = np.linalg.solve(U, y)
    return x


# Metode dekomposisi Crout
def metode_crout(A, b):
    n = len(A) 
    L = np.zeros((n, n)) 
    U = np.zeros((n, n)) 

    # Inisialisasi matriks segitiga bawah 'L'
    for i in range(n):
        L[i, i] = 1
    
    # Perhitungan matriks segitiga atas 'U' dan elemen-elemen 'U'
    for j in range(n):
        for i in range(j, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(j))
    
    # Perhitungan matriks segitiga bawah 'L' dan elemen-elemen 'L'
    for j in range(n):
        for i in range(j+1, n):
            L[i, j] = (A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))) / U[j, j]
    
    # Hitung solusi 'y' dan 'x'
    y = np.linalg.solve(L, b.astype(float))
    x = np.linalg.solve(U, y)
    return x

# Kode pengujian
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])
b = np.array([8, -11, -3])

print("Metode Matriks Balikan:")
print(metode_matriks_balikan(A, b))

print("\nMetode Dekomposisi LU Gauss:")
print(metode_lu_gauss(A, b))

print("\nMetode Dekomposisi Crout:")
print(metode_crout(A, b))
