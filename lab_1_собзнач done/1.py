import math

def f(x,A):
    return x**4 - x**3 * A[0][0] - x**2 * A[0][1] - x * A[0][2] - A[0][3] 

def scalar_multiply_vector(scalar, vector):
    res = []
    for i in range(len(vector)):
        res.append(scalar*vector[i])
    return res

def vectors_sub(vector1,vector2):
    res = []
    for i in range (len(vector1)):
        res.append(vector1[i] - vector2[i])
    return res

def scalar_multiply(matrix, scalar):
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    result = [[0 for _ in range(num_cols)] for _ in range(num_rows)]

    for i in range(num_rows):
        for j in range(num_cols):
            result[i][j] = matrix[i][j] * scalar

    return result

def determinant(matrix):
    if len(matrix) == len(matrix[0]):
        if len(matrix) == 1:
            return matrix[0][0]
        else:
            det = 0
            for col in range(len(matrix)):
                minor = [row[:col] + row[col+1:] for row in matrix[1:]]
                cofactor = (-1) ** col * determinant(minor)
                det += matrix[0][col] * cofactor
            return det
    else:
        raise ValueError("Матрица должна быть квадратной")

def transpose_matrix(matrix):
    rows, cols = len(matrix), len(matrix[0])
    transposed = [[0] * rows for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]

    return transposed

def invertMatrix(matrix):
    n = len(matrix)
    
    augmented_matrix = [[0.0 for _ in range(2 * n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            augmented_matrix[i][j] = matrix[i][j]
        augmented_matrix[i][i + n] = 1.0

    for i in range(n):
        divisor = augmented_matrix[i][i]
        for j in range(2 * n):
            augmented_matrix[i][j] /= divisor

        for k in range(n):
            if k != i:
                factor = augmented_matrix[k][i]
                for j in range(2 * n):
                    augmented_matrix[k][j] -= factor * augmented_matrix[i][j]

    inverse = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            inverse[i][j] = augmented_matrix[i][j + n]

    return inverse

def MatrixMatrixMult(matrix1, matrix2):
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Количество столбцов в первой матрице должно быть равно количеству строк во второй матрице")

    result = [[0] * len(matrix2[0]) for _ in range(len(matrix1))]

    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]

    return result

def MatrixVectorMult(matrix, vector):
    if len(matrix[0]) != len(vector):
        raise ValueError("Количество столбцов в матрице должно быть равно количеству элементов вектора")

    result = [0] * len(matrix)
    for i in range(len(matrix)):
        for j in range(len(vector)):
            result[i] += matrix[i][j] * vector[j]

    return result

def bisection(a, b, A, tol=1e-6):
    if f(a, A) * f(b, A) >= 0:
        return 0
    else:
        x = (a+b)/2
        while (abs(b - a) > tol):
            x = (a+b)/2
            if f(a, A) * f(x, A) < 0:
                b = x
            else:
                a = x
    return x

# Метод Данилевского
def method(A):
    N = len(A)
    # Первый этап
    B = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
            B[i][i] = 1
    for j in range(N):
        n = 3
        if(j != n-1):
            B[n-1][j] = -(A[n][j]) / (A[n][n-1])
        else:
            B[n-1][n-1] = 1 / (A[n][n-1])
    C = MatrixMatrixMult(A, B)
    A = MatrixMatrixMult(invertMatrix(B),C)
    B_f = B
    # Второй этап
    for i in range(n-2, -1, -1):
        B = [[0 for _ in range(N)] for _ in range(N)]
        for k in range(N):
            B[k][k] = 1
        for j in range(N):
            n = 3
            if(j != i):
                B[i][j] = -(A[i+1][j]) / (A[i+1][i])
            else:
                B[i][i] = 1 / (A[i+1][i])
        C = MatrixMatrixMult(A, B)
        A = MatrixMatrixMult(invertMatrix(B),C)
        B_f = MatrixMatrixMult(B_f, B)
    printMatrix(A, "Матрица Форбеннуса")

    # Поиск собственного значения методом бисекций
    print("\nСобственные значения\n")
    res = []
    for i in range(-100, 90):
        x = bisection(i, i+1, A)
        if x != 0:
            res.append(x)
            print(x)
    
    # поиск собственных векторов
    print("\nСобственные вектора\n")
    y = []
    for i in range (len(A)):
        tmp = [res[i]**3, res[i]**2, res[i]**1, 1]
        t = MatrixVectorMult(B_f, tmp)
        t = length(t)
        y.append(t)
        print(t)

    return A, res, y

def printMatrix(A, name):
    n = len(A)
    m = len(A[0])

    max_length = max(len(f"{A[i][j]:.6f}") for i in range(n) for j in range(m))

    name_length = len(name)
    name_indent = (max_length * m - name_length) // 2

    print(name.center(name_length + name_indent * 2))

    for i in range(n):
        for j in range(m):
            print(f"{A[i][j]:.6f}".rjust(max_length + 2), end=" ")
        print()

def length(v):
    res = []
    tmp = math.sqrt(sum(v[i] ** 2 for i in range(len(v))))
    for i in range (len(v)):
        res.append(v[i]/tmp)
    return res

def check(A,l,x):
    print("\nПроверка A*x - l*x = 0")
    for i in range (len(x)):
        res = vectors_sub(MatrixVectorMult(A, x[i]), scalar_multiply_vector(l[i],x[i]))
        print(res)


A = [[0, 1, 8, 1],
    [1, 10, 1, 0],
    [8, 1, 0, 1],
    [1, 0, 1, 0]]

N = 3

P = math.log(N)/math.log(10.0)
A[0][0] = N + 12
A[2][2] = -14 * N
A[3][3] = 3*(P + N)
# A = [[2.2, 1, 0.5, 2],
#      [1, 1.3, 2, 1],
#      [0.5, 2, 0.5, 1.6],
#      [2, 1, 1.6, 2]]

F, l, y = method(A)
check(A,l,y)


