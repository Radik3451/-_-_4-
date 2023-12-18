import numpy as np
import matplotlib.pyplot as plt
import math

# Ваши данные
H = np.array([i*15 for i in range(25)])
print(H)
B = np.array([55, 55.5, 42, 28, 16, 8, 5, 10, 19, 32, 46, 54, 56, 51, 41, 27, 15, 7.5, 6, 11, 21, 35, 47, 54, 56 ])

B2 = np.array([56*math.cos(math.radians(H[i]))**2 for i in range(25)])

print(math.radians(H[1]))

# Построение графика с аппроксимированной линией
plt.plot(H, B, marker='o', linestyle='-', color='b')
plt.plot(H, B2, color='r')
# Настройки графика
plt.xlabel('phi (Угол)')
plt.ylabel('I(phi) (Интенсивность)')
plt.legend()
plt.grid(True)
plt.show()