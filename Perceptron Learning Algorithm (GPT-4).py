# Data
E_list = [
    [1.7, 1, 4, -1],
    [5.5, 2, 9, +1],
    [2.2, 5, 6, +1],
    [1.3, 4, 5, +1],
    [1.4, 6, 0.7, -1],
    [4.2, 1, 1.5, -1]
]

# Learning Algorithm Function
def perceptron_learning_algorithm(E_list, learning_rate=0.1, max_iterations=1000):
    w = [0, 0, 0, 0] # w0 is the bias
    for iteration in range(max_iterations):
        error_count = 0
        for E in E_list:
            x0, x1, x2, x3, class_value = 1, E[0], E[1], E[2], E[3] # x0 = 1 for bias
            dot_product = w[0]*x0 + w[1]*x1 + w[2]*x2 + w[3]*x3
            prediction = 1 if dot_product >= 0 else -1
            
            if prediction != class_value:
                error_count += 1
                # Update weights
                w[0] += learning_rate * class_value * x0
                w[1] += learning_rate * class_value * x1
                w[2] += learning_rate * class_value * x2
                w[3] += learning_rate * class_value * x3
        
        if error_count == 0:
            # All examples classified correctly
            break
    
    return w, iteration + 1

# Train the perceptron
weights, iterations = perceptron_learning_algorithm(E_list)

print(f"Final weights: {weights}")
print(f"Converged after {iterations} iterations")
