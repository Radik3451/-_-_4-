import numpy as np
import math
import matplotlib.pyplot as plt

# Определение функции F(X) - система нелинейных уравнений
def f(X):
    a = 3
    f1 = X[0] - math.cos(X[1]) - 1
    f2 = X[1] - math.log(X[0] + 1) - a
    return [f1, f2]
    # f1 = X[0] + X[1] - 3
    # f2 = X[0]**2 + X[1]**2 - 9
    # return [f1, f2]

def gaus(A, B):
    n = len(B)
    
    # Прямой ход
    for i in range(n):
        # Поиск главного элемента в столбце
        max_row = i
        for j in range(i+1, n):
            if abs(A[j][i]) > abs(A[max_row][i]):
                max_row = j
        
        # Обмен строк, чтобы главный элемент был надоставлен вверх
        A[i], A[max_row] = A[max_row], A[i]
        B[i], B[max_row] = B[max_row], B[i]
        
        # Деление строки i на главный элемент
        pivot = A[i][i]
        A[i] = [x / pivot for x in A[i]]
        B[i] /= pivot
        
        # Вычитание строки i из остальных строк
        for j in range(n):
            if j != i:
                factor = A[j][i]
                A[j] = [A[j][k] - factor * A[i][k] for k in range(n)]
                B[j] -= factor * B[i]
    
    # Обратный ход
    X = np.zeros(n)
    for i in range(n-1, -1, -1):
        X[i] = B[i]
        for j in range(i+1, n):
            X[i] -= A[i][j] * X[j]
    
    return X

# Определение градиента функции F(X)
def gradient(X):
    a = 4
    df1_dx1 = 1
    df1_dx2 = math.sin(X[1])
    df2_dx1 = 1 / (X[0] + 1)
    df2_dx2 = 1
    # df1_dx1 = 1
    # df1_dx2 = 1
    # df2_dx1 = 2*X[0]
    # df2_dx2 = 2*X[1]
    W = [[df1_dx1, df1_dx2],
        [df2_dx1, df2_dx2]]
    return W

def vector_add(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Длины векторов должны быть одинаковыми")

    result = [x + y for x, y in zip(vector1, vector2)]
    return result

def norm(a):
    norm = 0
    for i in range(len(a)):
        norm += math.pow(a[i], 2)
    return math.sqrt(norm)

def nev(x):
    nev_vector = []
    for i in range (len(x)):
        nev_vector.append(f(x)[i] - 0)
    print(f"Вектор невязки: {nev_vector}")

    print(f"Норма вектора невязки: {norm(nev_vector)}")

# Начальное приближение
X = [1.0, 5.0]

iter = 1

# Для отображения изменений X на графике
X_history = [X]

grad = gradient(X)

f0 = f(X)
f0[0],f0[1] = -f0[0], -f0[1]
res = gaus(grad,f0)
X = vector_add(X, res)
eps = max(abs(res[0]), abs(res[1]))

while (eps > 1e-6):
    grad = gradient(X)

    f0 = f(X)
    f0[0],f0[1] = -f0[0], -f0[1]
    res = gaus(grad,f0)
    X = vector_add(X, res)
    eps = max(abs(res[0]), abs(res[1]))
    X_history.append(X)

    iter += 1

X_history = np.array(X_history)

print(f"Финальное значение X: {X}")
print(f"Количество итераций: {iter}")

nev(X)

# Визуализация изменений X на графике
plt.plot(X_history[:, 0], X_history[:, 1], marker='o')
plt.xlabel('X[0]')
plt.ylabel('X[1]')
plt.title('Изменение X на графике')
plt.grid(True)
plt.show()
