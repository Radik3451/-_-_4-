import math
import matplotlib.pyplot as plt

#Точное решение
def f(x):
    return math.tan(1/3*pow(x,3))

# Производная
def df(x, y):
    return pow(x,2) *(pow(y,2) + 1)

def runge_kutta(h, xi, yi):
    k1 = df(xi, yi)
    k2 = df(xi + h/3, yi + h*k1/3)
    k3 = df(xi + 2*h/3, yi + 2*k2*h/3)
    y_next = yi + (k1 + 3*k3) * h / 4
    return y_next

def find_norm(y, y1):
    eps = [0 for i in range(n)]
    for i in range (0,n):
        eps[i] = y[i] - y1[i]

    norm = 0
    for i in range (0,n):
        norm += eps[i]**2
    norm = math.sqrt(norm)
    
    return(norm)

h1 = 0.1
a = 0.0
b = 1.0
n = int((b-a)/h1) + 1
x = [a + i*h1 for i in range(n)]

answer = [f(xi) for xi in x]

# метод Рунге Кутта
y1 = [0 for i in range(n)]
y1[0] = 0.0
for i in range(0, 5):
    y1[i+1] = runge_kutta(h1, x[i], y1[i])

for i in range(5, n-1):
    # Предсказание (по Адамсу)
    predictor = y1[i] + h1 * (55/24 * df(x[i], y1[i]) - 59/24 * df(x[i-1], y1[i-1]) + 37/24 * df(x[i-2], y1[i-2]) - 3/8 * df(x[i-3], y1[i-3]))

    # Коррекция (по Адамсу)
    y1[i+1] = y1[i] + h1 * (9/24 * df(x[i+1], predictor) + 19/24 * df(x[i], y1[i]) - 5/24 * df(x[i-1], y1[i-1]) + 1/24 * df(x[i-2], y1[i-2]))

norm = find_norm(answer, y1)
print("\tРунге Кутта + Адамс\n")
print(answer)
print(y1)
print("Общая погрешность: ", norm, "\n")

# Уточнение методом Рунге
h2 = 0.2
p = 3
r = 2
y2 = [0 for i in range(n)]
y2[0] = 0.0
R = [0 for i in range(n)]
for i in range(0, 4, 2):
    y2[i+2] = runge_kutta(h2, x[i], y2[i])

for i in range(4, n-2, 2):
    # Предсказание (по Адамсу)
    predictor = y2[i] + h2 * (55/24 * df(x[i], y2[i]) - 59/24 * df(x[i-2], y2[i-2]) + 37/24 * df(x[i-4], y2[i-4]) - 3/8 * df(x[i-6], y2[i-6]))

    # Коррекция (по Адамсу)
    y2[i+2] = y2[i] + h2 * (9/24 * df(x[i+2], predictor) + 19/24 * df(x[i], y2[i]) - 5/24 * df(x[i-2], y2[i-2]) + 1/24 * df(x[i-4], y2[i-4]))

for i in range(2, 5, 2):
    R[i] = (y1[i]-y2[i])/(math.pow(r, p) - 1)
    y2[i] = y1[i] + R[i]
p = 4
for i in range(6, n, 2):
    R[i] = (y1[i]-y2[i])/(math.pow(r, p) - 1)
    y2[i] = y1[i] + R[i]
for i in range(1,n,2):
    R[i] = (R[i+1] + R[i-1])/2
    y2[i] = y1[i] + R[i]

print("\tУточнение методом Рунге\n")
norm = find_norm(answer, y2)
print(answer)
print(y1)
print("Общая погрешность: ", norm, "\n")

plt.plot(x, y1, label='Рунге-Кутты + Адамс')
plt.plot(x, y2, label='Уточнение методом Рунге')
plt.plot(x, answer, label='Точное решение')
plt.legend()
plt.show()
