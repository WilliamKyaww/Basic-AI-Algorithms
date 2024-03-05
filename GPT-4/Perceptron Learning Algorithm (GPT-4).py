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
def perceptron_learning_algorithm(E_list, max_iterations=1000):
    w = [0, 0, 0, 0]  # Initialize weights w0 (bias), w1, w2, w3
    for iteration in range(max_iterations):
        error_count = 0
        for E in E_list:
            x1, x2, x3, class_value = E[0], E[1], E[2], E[3]
            # Compute the dot product, considering w[0] as bias
            dot_product = w[0] + (w[1] * x1) + (w[2] * x2) + (w[3] * x3)
            prediction = 1 if dot_product >= 0 else -1

            # Update weights if the prediction is wrong
            if prediction != class_value:
                error_count += 1
                w[0] += class_value  # Update bias
                w[1] += class_value * x1
                w[2] += class_value * x2
                w[3] += class_value * x3

                # Print weights after each update in the specified format
                print(f"Iteration {iteration + 1}, Example {E_list.index(E) + 1}: w.x = {w[0]} + ({w[1]})x1 + ({w[2]})x2 + ({w[3]})x3")
        
        # If no errors, learning is complete
        if error_count == 0:
            print("All examples classified correctly.")
            break

    # Return final weights and the number of iterations it took
    return w, iteration + 1

# Train the perceptron and print final weights
weights, iterations = perceptron_learning_algorithm(E_list)
final_weights_format = f"Final weights: w.x = {weights[0]} + ({weights[1]})x1 + ({weights[2]})x2 + ({weights[3]})x3"
print(final_weights_format)
print(f"Converged after {iterations} iterations")
