# Data
E_list = [
    [1, 4, -1],  # E1
    [2, 9, +1],  # E2
    [5, 6, +1],  # E3
    [4, 5, +1],  # E4
    [6, 0.7, -1],# E5
    [1, 1.5, -1] # E6
]

# Adaline Learning Algorithm Function
def adaline_learning_algorithm(learning_rate, x1, x2, class_value, w0, w1, w2):
    # Calculate the dot product
    dotProduct = w0 + (w1 * x1) + (w2 * x2)
    
    # Update weights based on the error (class_value - dotProduct)
    w0 += learning_rate * (class_value - dotProduct)
    w1 += learning_rate * (class_value - dotProduct) * x1
    w2 += learning_rate * (class_value - dotProduct) * x2

    return w0, w1, w2, f"w.x = {w0} + ({w1})x1 + ({w2})x2"

# Initial weights
w0, w1, w2 = 1, 2, 3

# Initialize learning rate
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

print("\nWeights have stabilized.")
