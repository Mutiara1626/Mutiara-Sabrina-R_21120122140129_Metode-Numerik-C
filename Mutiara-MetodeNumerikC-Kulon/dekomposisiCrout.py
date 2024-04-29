import numpy as np

# Metode dekomposisi Crout
def metode_crout(A, b):
    n = len(A) # Menghitung ukuran matriks 'A' (jumlah baris atau kolom)
    L = np.zeros((n, n)) # matriks segitiga bawah
    U = np.zeros((n, n)) # matriks segitiga atas

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

print("\nMetode Dekomposisi Crout:")
print(metode_crout(A, b))
