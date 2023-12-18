import math
import matplotlib.pyplot as plt

def f(x):
    return math.log(-1 + pow(math.e, pow(math.e,x)))

def df(x, y):
    return pow(math.e, x - y) + pow(math.e, x)

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


h1 = 0.1
a = 0.0
b = 1.0
n = int((b-a)/h1) + 1
x = [a + i*h1 for i in range(n)]

answer = [f(xi) for xi in x]

# Явный метод Эйлера
y = [0.541325]
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

# Метод Хойна
y1 = [0 for i in range (n)]
y1[0] = 0.541325
for i in range(0, n-1):
    y1[i+1] = y1[i] + h1/2*(df(x[i+1],y[i+1]) + df(x[i],y1[i]))

eps = [0 for i in range(n)]
for i in range (0,n):
    eps[i] = y1[i] - answer[i]

norm = 0
for i in range (0,n):
    norm += eps[i]**2
norm = math.sqrt(norm)
print("\tМетод Хойна\n")
PrintVectors(x, y1, answer, eps)
print("Общая погрешность: ", norm, "\n")


# уточнение методом Рунге
h2 = 0.2
r = h2/h1
p = 2

y2 = [0 for i in range(n)]
R = [0.0 for i in range(n)]
y2[0] = 0.541325
for i in range(0, n-2, 2):
    y2[i+2] = y1[i] + h2/2*(df(x[i+2],y[i+2]) + df(x[i],y[i]))
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
plt.plot(x, y1, label='Метод Хойна')
plt.plot(x, y2, label='Уточнение методом Рунге с h = 0.2')
plt.legend()
plt.show()
