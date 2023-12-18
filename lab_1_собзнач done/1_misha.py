import math

def f(x,A):
    return x**4 - x**3 * A[0] - x**2 * A[1] - x * A[2] - A[3] 

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

def vector_add(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Длины векторов должны быть одинаковыми")

    result = [x + y for x, y in zip(vector1, vector2)]
    return result

def matrix_sub(matrix1, matrix2):
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        raise ValueError("Матрицы должны иметь одинаковый размер")
    
    result = []
    for i in range(len(matrix1)):
        row = []
        for j in range(len(matrix1[0])):
            row.append(matrix1[i][j] - matrix2[i][j])
        result.append(row)
    
    return result

def scalar_multiply(matrix, scalar):
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    result = [[0 for _ in range(num_cols)] for _ in range(num_rows)]

    for i in range(num_rows):
        for j in range(num_cols):
            result[i][j] = matrix[i][j] * scalar

    return result

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

# Метод Лаверрье
def method(A):
    N = len(A)
    A_main = A
    E = [[0 for _ in range(N)] for _ in range(N)]
    p_list = []
    B_list = [[] for i in range (N)]

    for i in range(0,N):
        E[i][i] = 1

    for k in range(1,N):
        p = 0
        for i in range(N):
            p += A[i][i]
        p = (1/k) * p
        p_list.append(p)

        B = matrix_sub(A, scalar_multiply(E, p))
        for i in range(len(B)):
            B_list[k].append(B[i][0])
        A = MatrixMatrixMult(A_main, B)

    p = 0
    for i in range(N):
        p += A[i][i]
    p = (1/N) * p
    p_list.append(p)

    A_inv = scalar_multiply(B, 1/p)

    print("\nСобственные значения\n")
    res = []
    for i in range(-500, 500):
        x = bisection(i, i+1, p_list)
        if x != 0:
            res.append(x)
            print(x)
    
    y0 = [1,0,0,0]

    print("\nСобственные вектора\n")
    y = []
    for i in range(N):
        y1 = vector_add(scalar_multiply_vector(res[i], y0), B_list[1])
        y2 = vector_add(scalar_multiply_vector(res[i], y1), B_list[2])
        y3 = vector_add(scalar_multiply_vector(res[i], y2), B_list[3])
        t = length(y3)
        print(t)
        y.append(t)

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

N = 6

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