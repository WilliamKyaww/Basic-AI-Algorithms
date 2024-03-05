import matplotlib.pyplot as plt
import numpy as np

# Data
E1 = [1, 4, -1]
E2 = [2, 9, +1]
E3 = [5, 6, +1]
E4 = [4, 5, +1]
E5 = [6, 0.7, -1]
E6 = [1, 1.5, -1]
E_list = [E1, E2, E3, E4, E5, E6]

# Adaline -----------------------------------------------------------------------------------------------------------------

print("Adaline learning algorithm:")

# Adaline Learning Algorithm Function 
def adaline_learning_algorithm(learning_rate, x1, x2, class_value, w0, w1, w2):
    dotProduct = w0 + (w1*x1) + (w2*x2)

    w0 += learning_rate * (class_value - dotProduct)
    w1 += learning_rate * (class_value - dotProduct) * x1
    w2 += learning_rate * (class_value - dotProduct) * x2

    return w0, w1, w2, f"w.x = {w0} + ({w1})x1 + ({w2})x2"

# Initialise weights
w0, w1, w2 = 1, 2, 3

# Initialise learning rate
learning_rate = 0.01

# Iterate until weights become stable
weights_changed = True
iteration = 0
while weights_changed:
    weights_before = [w0, w1, w2]
    for i, E in enumerate(E_list):
        x1, x2, class_value = E[0], E[1], E[2]
        w0, w1, w2, equation = adaline_learning_algorithm(learning_rate, x1, x2, class_value, w0, w1, w2)
        print(f"Iteration {iteration + 1}, Row {i+1}: {equation}")
    weights_after = [w0, w1, w2]
    
    if weights_before == weights_after:
        weights_changed = False
    else:
        iteration += 1

print("Weights have stabilised.\n")

# Adaline Plot -----------------------------------------------------------------------------------------------------------------

print("Perceptron learning algorithm:")

# Get x and y values
x_values = [E[1] for E in E_list]
y_values = [E[0] for E in E_list]

final_w0, final_w1, final_w2 = w0, w1, w2

# Prepare dataset for plotting
class_1_x1 = [E[0] for E in E_list if E[2] == 1]
class_1_x2 = [E[1] for E in E_list if E[2] == 1]
class_minus_1_x1 = [E[0] for E in E_list if E[2] == -1]
class_minus_1_x2 = [E[1] for E in E_list if E[2] == -1]

# Plot data points
plt.scatter(class_1_x1, class_1_x2, color='blue', label='Class +1')
plt.scatter(class_minus_1_x1, class_minus_1_x2, color='red', label='Class -1')

# Calculate and plot the decision boundary: w0 + w1*x1 + w2*x2 = 0 -> x2 = (-w0 - w1*x1) / w2
x1_values = np.linspace(min(x_values), max(x_values), 100)
x2_values = (-final_w0 - final_w1*x1_values) / final_w2
plt.plot(x1_values, x2_values, 'g--', label='Adaline Decision Boundary')

# Perceptron -----------------------------------------------------------------------------------------------------------------

# Perceptron learning algorithm function 
def perceptron_learning_algorithm(x1, x2, class_value, w0, w1, w2):
    classification = False
    while not classification:
        dotProduct = w0 + (w1*x1) + (w2*x2) 
        if class_value == 1 and dotProduct <= 0:
            w0 += class_value
            w1 += class_value * x1
            w2 += class_value * x2
        elif class_value == -1 and dotProduct >= 0:
            w0 += class_value
            w1 += class_value * x1
            w2 += class_value * x2
        else:
            classification = True
    return w0, w1, w2, f"w.x = {w0} + {w1}x₁ + {w2}x₂"

# Initial weights
w0, w1, w2 = 0, 0, 0

# Iterate until weights become stable
weights_changed = True
iteration = 0
while weights_changed:
    weights_before = [w0, w1, w2]
    for i, E in enumerate(E_list):
        x1, x2, class_value = E[0], E[1], E[2]
        w0, w1, w2, equation = perceptron_learning_algorithm(x1, x2, class_value, w0, w1, w2)
        print(f"Iteration {iteration + 1}, Row {i+1}: {equation}")
    weights_after = [w0, w1, w2]
    
    if weights_before == weights_after:
        weights_changed = False
    iteration += 1

print("Weights have stabilised.")

# Perceptron Plot -----------------------------------------------------------------------------------------------------------------

final_w0, final_w1, final_w2 = w0, w1, w2

# Calculate and plot the decision boundary: w0 + w1*x1 + w2*x2 = 0 -> x2 = (-w0 - w1*x1) / w2
x1_values = np.linspace(min(x_values), max(x_values), 100)
x2_values = (-final_w0 - final_w1*x1_values) / final_w2
plt.plot(x1_values, x2_values, 'b--', label='Perceptron Decision Boundary')

plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Learning Algorithms - Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()




