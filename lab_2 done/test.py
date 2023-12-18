def f1(x, y):
    return (x**2 + 9) * y - 27
def f2(x, y):
    return (x - 1.5)**2 + (27/(y**2 + 9) - 1.5)**2 - 9
# Метод простых итераций
def simple_iteration_method(x0, y0, epsilon, max_iter):
    for _ in range(max_iter):        
        y = 27 / (x0**2 + 9)
        x = (9 - (y - 1.5)**2)**0.5 + 1.5
        if abs(x - x0) < epsilon and abs(y - y0) < epsilon:
            return x, y        
        x0, y0 = x, y
    return None, None
# Начальные приближения и точность
x0 = -1.3
y0 = 2.5
epsilon = 0.000001
max_iterations = 100000
# Вызов метода простых итераций
solution_x, solution_y = simple_iteration_method(x0, y0, epsilon, max_iterations)
# Вывод результата
print("Solution:")
print("x =", solution_x)
print("y =", solution_y)
