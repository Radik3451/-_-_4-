import math
import matplotlib.pyplot as plt

def f(x):
    return (x - math.tan((-1+x)/x)) / math.pow(x, 2)

def df(x, y):
    return -math.pow(y, 2) - 1/(math.pow(x, 4))

def PrintVectors(x, y, answer, eps):
    print("    x    |     y    |  answer  |   eps   ")
    for i in range(len(x)):
        print(f"{x[i]:.6f} | {y[i]:.6f} | {answer[i]:.6f} | {eps[i]:.6f}")
    print()

def PrintVectors2(x, y, y2, eps):
    print("    x    |   method |   runge  |   eps   ")
    for i in range(len(x)):
        print(f"{x[i]:.6f} | {y[i]:.6f} | {y2[i]:.6f} | {eps[i]:.6f}")
    print()

def func(yn1, tn1, yn, tn, tau):
    # Здесь определите вашу функцию
    # Ваше уравнение: yn1 - 0.5 * tau * f(tn1, yn1) - yn - 0.5 * tau * f(tn, yn)
    return yn1 - 0.5 * tau * df(tn1, yn1) - yn - 0.5 * tau * df(tn, yn)

def derivative(yn1, tn1, tau):
    # Здесь определите производную вашей функции по yn1
    # Пример: return 1 - 0.5 * tau * df/dy(tn1, yn1)
    return 1 - 0.5 * tau * df(tn1, yn1)

def newton_method(initial_guess, tn1, yn, tn, tau, tol=1e-6, max_iter=100):
    guess = initial_guess
    iter_count = 0

    while abs(func(guess, tn1, yn, tn, tau)) > tol and iter_count < max_iter:
        guess = guess - func(guess, tn1, yn, tn, tau) / derivative(guess, tn1, tau)
        iter_count += 1

    return guess
h1 = 0.1
a = 1.0
b = 2.0
n = int((b-a)/h1) + 1
x = [a + i*h1 for i in range(n)]

answer = [f(xi) for xi in x]

#метод Трапеций
y = [1]
for i in range(1, n):
    y.append(y[i-1] + h1*df(x[i-1], y[i-1]))

eps = [0 for i in range(n)]
for i in range (0,n):
    eps[i] = y[i] - answer[i]

norm = 0
for i in range (0,n):
    norm += eps[i]**2
norm = math.sqrt(norm)
print("\t\tЯвный метод Эйлера\n")
PrintVectors(x, y, answer, eps)
print("Общая погрешность: ", norm, "\n")

# Метод трапеций
y1 = [0 for i in range (n)]
y1[0] = 1.0
for i in range(0, n-1):
    y1[i+1] = newton_method(0.1, x[i+1], y1[i], x[i], h1)
    # y1[i+1] = y1[i] + 0.5*h1*(df(x[i],y1[i]) + df(x[i+1],y[i+1]))

eps = [0 for i in range(n)]
for i in range (0,n):
    eps[i] = y1[i] - answer[i]

norm = 0
for i in range (0,n):
    norm += eps[i]**2
norm = math.sqrt(norm)
print("\tМетод трапеций\n")
PrintVectors(x, y1, answer, eps)
print("Общая погрешность: ", norm, "\n")


# уточнение методом Рунге
h2 = 0.2
r = h2/h1
p = 2

y2 = [0 for i in range(n)]
R = [0.0 for i in range(n)]
y2[0] = 1.0
for i in range(0, n-2, 2):
    y2[i+2] = newton_method(0.1, x[i+2], y2[i], x[i], h2)
for i in range(2, n, 2):
    R[i] = (y1[i]-y2[i])/(math.pow(r, p) - 1)
    y2[i] = y1[i] + R[i]
for i in range(1,n,2):
    R[i] = (R[i+1] + R[i-1])/2
    y2[i] = y1[i] + R[i]

for i in range (0,n):
    eps[i] = y2[i] - answer[i]

norm = 0
for i in range (0,n):
    norm += eps[i]**2
norm = math.sqrt(norm)
print("\tУточнение методом Рунге\n")
PrintVectors(x, y2, answer, eps)
print("Общая погрешность: ", norm, "\n")

plt.plot(x, y, label='Явный метод Эйлера')
plt.plot(x, answer, label='Точное решение')
plt.plot(x, y1, label='Предиктор-корректор')
plt.plot(x, y2, label='Уточнение методом Рунге с h = 0.2')
plt.legend()
plt.show()
