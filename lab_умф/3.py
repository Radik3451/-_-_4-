from copy import deepcopy
import math


def f(x, t):
    return m_alpha*(x**2 - 2*t)

def matrix_matrix_mult(matrix1, matrix2):
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Количество столбцов в первой матрице должно быть равно количеству строк во второй матрице")

    result = [[0] * len(matrix2[0]) for _ in range(len(matrix1))]

    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]

    return result

def print_matrix(A, name):
    nx = len(A)
    m = len(A[0])
    print(f'\t\t\t{name}')
    for i in range(nx-1, -1, -1):
        for j in range (m):
            print(f"{A[i][j]:5.3f}", end=' ')
        print()

def implicit_scheme(matrix):
    tmp = (tau)/(h**2)
    A = [0 for i in range(nx)]
    B = [0 for i in range(nx)]
    C = [0 for i in range(nx)]
    
    # T = [[0 for i in range(nx)] for i in range(nx)]
    for i in range(nx):
        if(i != nx-1):
            B[i] = 2*tmp + 1

        if(i < nx-1):
            C[i] = -tmp
        else: 
            C[i] = 0

        if(i > 0):
            A[i] = -tmp
        else: 
            A[i] = 0
    B[0] = 1
    B[nx-1] = 1
    C,A = A,C

    # tridiagonal=[[0 for _ in range(nx)] for _ in range(nx)]
    # for i in range(nx):
    #     tridiagonal[i][i] = B[i]
        
    #     if(i > 0):
    #         tridiagonal[i][i-1] = A[i]
        
    #     if(i < nx-1):
    #         tridiagonal[i][i+1] = C[i]

    for k in range (1,nt):
        d = [matrix[k-1][i] + tau*f(x[i],t[k-1]) for i in range(nx)]
        # d = [matrix[k-1][i] for i in range(nx)]
        # Прямой ход метода прогонки
        alpha = [0 for _ in range(nx)]
        beta = [0 for _ in range(nx)]

        alpha[0] = -C[0]/B[0]
        beta[0] = d[0]/B[0]

        for i in range(1, nx-1):
            alpha[i] = C[i]/(-B[i] - A[i]*alpha[i-1])
            # beta[i] = (A[i]*beta[i-1] - f[i])/(-B[i] - A[i]*alpha[i-1])
            beta[i] = (d[i] - A[i]*beta[i-1])/(B[i] + A[i]*alpha[i-1])
        beta[nx-1] = (A[nx-1]*beta[nx-2] - d[nx-1])/(-B[nx-1] - A[nx-1]*alpha[nx-2])

        # Обратный ход метода прогонки
        U = [0 for i in range(nx)]
        U[nx-1] = beta[nx-1]
        for i in range(nx-2,-1,-1):
            U[i] = alpha[i]*U[i+1] + beta[i]

        for i in range (1,nx-1):
            matrix[k][i] = U[i]

    print_matrix(matrix, "Неявная схема")
    return matrix

def obvious_scheme(matrix):
    for i in range(1,nt):
        for j in range (1,nx-1):
            # print(matrix[i-1][j], matrix[i-1][j+1] - 2*matrix[i-1][j] + matrix[i-1][j-1]))
            matrix[i][j] = matrix[i-1][j] + (tau/h**2)*(matrix[i-1][j+1] - 2*matrix[i-1][j] + matrix[i-1][j-1]) + tau*f(x[j],t[i-1])
    print_matrix(matrix, "Явная схема")

def check(matrix1, matrix2):
    res = [[0 for _ in range(nx)] for _ in range(nt)]
    for i in range(nt):
        for j in range(nx):
            res[i][j] = matrix1[i][j] - matrix2[i][j]
    
    norm = 0.0
    for i in range(nt):
        for j in range(nx):
            norm += res[i][j]**2

    norm = math.sqrt(norm)

    return res, norm


nx = 3
m_alpha = 0.5*nx

# Шаги по x и t
h = 0.1
tau = 0.01

# Границы значений x и t
ax = 0
bx = 1
at = 0
bt = 0.05

# Количество отрезков x и t
nx = int((bx-ax)/h+1)
nt = int((bt-at)/tau+1)

# Заполнение значений x и t
x = [i*h for i in range(nx)]
t = [i*tau for i in range(nt)]

A = [[0 for _ in range(nx)] for _ in range(nt)]
for i in range (nt):
    for j in range (nx):
        if(j == 0):
            A[i][j] = 0
        elif(j == nx-1):
            A[i][j] = m_alpha*t[i]
        elif(i == 0):
            A[i][j] = 0
        else: A[i][j] = 1
# for i in range (nt):
#     for j in range (nx):
#         if(j == 0):
#             A[i][j] = 0
#         elif(j == nx-1):
#             A[i][j] = 0
#         elif(i == 0):
#             A[i][j] = 4*x[j]*(1 - x[j])
#         else: A[i][j] = 1

print_matrix(A, 'Начальная матрица')
print('\n')

start_A = deepcopy(A)
# Неявная формула
A = implicit_scheme(A)
print('\n')

# Явная формула
obvious_scheme(start_A)

print("\n")
q = (1 - 4*(tau/h**2)*(math.sin(h*f(x[0],t[0]))/2)**2)
if (abs(q) <= 1):
    print("Явная схема устойчива")
else: print("Явная схема неустойчива")

print("\n")
q = (1 + 4*(tau/h**2)*(math.sin(h*f(x[0],t[0]))/2)**2)**(-1)
if (abs(q) <= 1):
    print("Неявная схема устойчива")
else: print("Неявная схема неустойчива")

res, norm = check(A, start_A)
print("\n")
print_matrix(res, 'Невязка')
print(f'Норма невязки: {norm}')