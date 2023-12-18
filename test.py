import numpy as np
import matplotlib.pyplot as plt

# Ваши данные
H = np.array([62.4, 46.8, 37.1, 33.2, 27.3])
B = np.array([74.8, 63.1, 72.4, 49.1, 11.7])

# Аппроксимация прямой линией
coefficients = np.polyfit(H, B, 1)
a, b = coefficients

# Построение графика с аппроксимированной линией
plt.plot(H, B, marker='o', linestyle='-', color='b', label='Экспериментальные данные')
plt.plot(H, a * H + b, linestyle='--', color='r', label=f'Аппроксимация: B = {a:.2f}H + {b:.2f}')

# Настройки графика
plt.title('Кривая намагничивания и аппроксимация')
plt.xlabel('H (напряженность магнитного поля)')
plt.ylabel('B (магнитная индукция)')
plt.legend()
plt.grid(True)
plt.show()

# Вычисление магнитной проницаемости и построение графика μ=f(H)
mu_0 = 4 * np.pi * 1e-7  # магнитная постоянная

# Вычисление магнитной проницаемости для каждого значения H
mu_values = (B - b) / (a * mu_0 * H)

# Построение графика μ=f(H)
plt.plot(H, mu_values, marker='o', linestyle='-', color='g')

# Настройки графика
plt.title('Зависимость μ от H')
plt.xlabel('H (напряженность магнитного поля)')
plt.ylabel('μ (магнитная проницаемость)')

# Отображение графика
plt.grid(True)
plt.show()

# Вывод магнитной проницаемости для каждого значения H
for i, h in enumerate(H):
    print(f'При H = {h:.2f}, магнитная проницаемость μ = {mu_values[i]:.2e} H/m')
