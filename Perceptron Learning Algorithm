import matplotlib.pyplot as plt
import numpy as np

# Assuming these are the final weights after training
w0 = 1
w1 = 2
w2 = -3

# Data Points: [x1, x2, class_value]
E_list = [
    [1, 4, -1],  # E1
    [2, 9, +1],  # E2
    [5, 6, +1],  # E3
    [4, 5, +1],  # E4
    [6, 0.7, -1],# E5
    [1, 1.5, -1] # E6
]

# Separate the data points based on their class
class_1 = np.array([e[:2] for e in E_list if e[2] == 1])
class_minus_1 = np.array([e[:2] for e in E_list if e[2] == -1])

# Plot the data points
plt.scatter(class_1[:, 0], class_1[:, 1], color='blue', label='Class +1')
plt.scatter(class_minus_1[:, 0], class_minus_1[:, 1], color='red', label='Class -1')

# Plot the decision boundary
x_values = np.linspace(0, 7, 100)  # Generate 100 points between 0 and 7 for x1
y_values = (-w0 - w1*x_values) / w2  # Solve for x2 in the equation w0 + w1*x1 + w2*x2 = 0
plt.plot(x_values, y_values, label='Decision Boundary')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Adaline Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()
