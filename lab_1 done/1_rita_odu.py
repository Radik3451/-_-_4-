import math
import matplotlib.pyplot as plt

#Точное решение
def f(x):
    return math.cos(x) - 1 + pow(math.e, -math.cos(x))

# Производная
def df(x, y):
    return y*math.sin(x) - math.cos(x)*math.sin(x)

def find_norm(y, y1):
    eps = [0 for i in range(n)]
    for i in range (0,n):
        eps[i] = y[i] - y1[i]

    norm = 0
    for i in range (0,n):
        norm += eps[i]**2
    norm = math.sqrt(norm)
    
    return(norm)

a = 0.0
b = 1.0
h1 = 0.1
y0 = 0.367879
n = int((b - a)/h1)+1
x = [i*h1 for i in range(n)]

# Точное решение
y=[f(x[i]) for i in range(n)]


# Явный метод Эйлера
y1 = [y0]
for i in range(n-1):
    y1.append(y1[i] + h1*df(x[i], y1[i]))
norm = find_norm(y, y1)

print("Явный метод Эйлера")
print(y)
print(y1)
print(norm)

# Неявный метод Эйлера
y2 = [y0]
for i in range(n-1):
    y2.append(y2[i] + h1*df(x[i+1], y1[i+1]))
norm = find_norm(y, y2)

print("Неявный метод Эйлера")
print(y)
print(y2)
print(norm)

# Уточнение методом Рунге
h2 = 0.2
r = h2/h1
p = 1

y3 = [0 for i in range(n)]
R = [0.0 for i in range(n)]
y3[0] = y0
for i in range(0, n-2, 2):
    y3[i+2] = y3[i] + h2*df(x[i+2], y1[i+2])
for i in range(2, n, 2):
    R[i] = (y2[i]-y3[i])/(math.pow(r, p) - 1)
    y3[i] = y2[i] + R[i]
for i in range(1,n,2):
    R[i] = (R[i+1] + R[i-1])/2
    y3[i] = y2[i] + R[i]
norm = find_norm(y, y3)

print("Уточнение методом Рунге")
print(y)
print(y3)
print(norm)

plt.plot(x, y, label='Точное решение')
plt.plot(x, y1, label='Явный метод Эйлера')
plt.plot(x, y2, label='Неявный метод')
plt.plot(x, y3, label='Уточнение методом Рунге с h = 0.2')
plt.legend()
plt.show()