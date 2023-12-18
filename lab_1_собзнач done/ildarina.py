import numpy as np
from sympy import *
import math

def f(x): 
    return (3*x*x-6*x+5+34*abs(x-6)) 

def euclidean_norm(vector):
    """
    Вычисляет евклидову норму вектора.

    Параметры:
    vector (list): Входной вектор.

    Возвращает:
    float: Евклидова норма вектора.
    """
    squared_sum = sum(x**2 for x in vector)
    return math.sqrt(squared_sum)

def check(A,l,x):
    """
    Проверка найденных собственных значений и векторов

    Параметры:
    A - исходная матрица
    l - собственные значения
    x - собственные вектора

    Возвращает разницу A*x - l*x
    """
    print("\nПроверка A*x - l*x = 0")
    for i in range (len(x)):
        # res = vectors_sub(MatrixVectorMult(A, x[i]), scalar_multiply_vector(l[i],x[i]))
        res = np.dot(A,x[i]) - l[i]*x[i]
        print(res)
    
Исходная матрица A
A = np.array([[0, 1, 8, 1],
              [1, 10, 1, 0],
              [8, 1, 0, 1],
              [1, 0, 1, 0]])

# Изменения в матрице А
N = 2
n = 4
P = np.log(N) / np.log(10.0)
print(P)

A[0,0] = N + 12
A[2,2] = -14 * N
A[3,3] = 3 * (P + N)
# Матрица А для тестирования(та, что в примере в лк)
# A = np.array([[2.2, 1, 0.5, 2],
#               [1, 1.3, 2, 1],
#               [0.5, 2, 0.5, 1.6],
#               [2, 1, 1.6, 2]])
print(A)
c = []
c.append([1, 0, 0, 0]) # начальный с

for i in range(1, 5):
 c.append(np.dot(A, c[i - 1])) # рекурсивно вычисляем c
C = np.array(c) # копия С
print ("Матрица для вычисления собств значений")
print(C)
cn = c.pop() # выделяем столбец свободных членов 
print ("столбец свободных членов")
print (cn)

c = np.array(c).transpose()

for i in range(4): # транспонируем матрицу коэффициентов C
 c[i] = list(reversed(c[i]))
print ("матрица для системы ур")
print (c)


p = np.linalg.solve(c, cn) 
print ("Коэффциенты")
print(p)

x = Symbol('x') # вычисляем собственные значения

L = solve(x**4 - p[0] * x**3 - p[1] * x**2 - p[2] * x - p[3], x)
real_parts = [re(sol) for sol in L]
print ("Собственные значения")
print(real_parts)

# l = max(real_parts) # максимальное собственное
# x=[]
# b=np.array([1, 0, 0, 0])
# for i in range(1, n): # находим коэффиценты ß
#     b[i] = b[i - 1] * l - p[i - 1]
# x = np.sum([b[i] * C[n - i - 1] for i in range(n)], axis=0)
# print ("Собственный вектор для максимального собств значения")
# print(x)

x=[]
for i in range(len(A)):
    q = [1]
    for j in range(1, len(A)):
        # print(real_parts[i])
        # print(q[j-1])
        # print(p[j-1])
        q.append(real_parts[i]*q[j-1] - p[j-1])
    x.append(C[3] + q[1]*C[2] + q[2]*C[1] + q[3]*C[0])
    # print(C[3] + q[1]*C[2] + q[2]*C[1] + q[3]*C[0])

print ("Собственный векторы")
for i in range(len(x)):
    # print(x[i])
    # print(euclidean_norm(x[i]))
    x[i] = x[i]/euclidean_norm(x[i])
    print(x[i])

check(A, real_parts, x)

# eigenvalues, eigenvectors = np.linalg.eig(A)
# print()
# print("Для проверки")
# print("Собственные значения:")
# print(eigenvalues)
# print("Собственные векторы:")
# print(eigenvectors)