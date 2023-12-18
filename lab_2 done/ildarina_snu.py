import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


# Определение функции F(X) - система нелинейных уравнений
def f(X):
    a = 3
    f1 = (X[0]**2 + a**2) * X[1] - a**3
    f2 = (X[0] - a/2)**2 + (X[1] - a/2)**2 - a**2
    # f1 = 0.1*X[0]**2 + X[0] + 0.2*X[1]**2 - 0.3
    # f2 = 0.2*X[0]**2 + X[1] - 0.1*X[0]*X[1] - 0.7
    return[f1, f2]

def func(X):
    a = 3
    f2 = a**3/(pow(X[0], 2) + pow(a, 2))
    f1 = math.sqrt(a**2 - (X[1]-a/2)**2 + X[0]*a - (a**2)/4)

    # f1 = math.sqrt(a**3/X[1] - a**2)
    # f2 = math.sqrt(a**2 - (X[0]-a/2)**2 + X[1]*a - (a**2)/4)

    # f1 = 0.3 - 0.1*X[0]**2 - 0.2*X[1]**2
    # f2 = 0.7 - 0.2*X[0]**2 + 0.1*X[0]*X[1]
    return [f1, f2]


def find_first_X():
    # Задаем диапазон значений для x и y
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)

    # Создаем сетку значений для x и y
    X, Y = np.meshgrid(x, y)

    # Вычисляем значения функций для каждой точки на сетке
    Z1, Z2 = f([X, Y])

    # Строим графики
    plt.figure(figsize=(8, 6))

    # График для f1
    plt.contour(X, Y, Z1, levels=[0], colors='r')

    # График для f2
    plt.contour(X, Y, Z2, levels=[0], colors='b')

    # Настройка графика
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title('Графики функций $f_1$ и $f_2$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def simple_iter_method(X):
    X_prev = X
    X_next = func(X)
    X_history.append(X_next)
    iter = 1
    print(f'{iter}\t{X_prev[0]}\t{X_prev[1]}\t{X_next[0]}\t{X_next[1]}')
    # print(max(abs(X_next[0] - X_prev[0]), abs(X_next[1] - X_prev[1])))
    while(max(abs(X_next[0] - X_prev[0]), abs(X_next[1] - X_prev[1])) > epsilon):
        X_prev = X_next
        X_next = func(X_next)
        X_history.append(X_next)
        iter += 1
        print(f'{iter}\t{X_prev[0]}\t{X_prev[1]}\t{X_next[0]}\t{X_next[1]}')
    return X_next, iter

def check(answer):
    nev = []
    for i in range(len(answer)):
        nev.append(f(answer)[i] - 0)
    print(nev)


# Точность ε=10^-6
epsilon = 1e-6
iter = 1


# Объединяем начальные приближения в один вектор
find_first_X()
input_string = input("Введите координаты пересечения через пробел: ")

# Разделяем строку по пробелу и преобразуем каждую часть в число
numbers = [float(x) for x in input_string.split()]
X = numbers
# X = [0.25, 0.75]
print(X)

X_history = [X]

# Запуск метода простых итераций
X_f, iter = simple_iter_method(X)
X_history = np.array(X_history)

print(f"Финальное значение X: {X_f}")
print(f"Количество итераций: {iter}")

check(X_f)

# Визуализация изменений X на графике
plt.plot(X_history[:, 0], X_history[:, 1], marker='o')
plt.xlabel('X[0]')
plt.ylabel('X[1]')
plt.title('Изменение X на графике')
plt.grid(True)
plt.show()
