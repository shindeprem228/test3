import numpy as np
import matplotlib.pyplot as plt


def function(x):
    return (x + 3) ** 2


def derivative(x):
    return 2 * (x + 3)


def gradient_descent(starting_x, learning_rate, num_iterations):
    x = starting_x
    x_values = [x]  

    for i in range(num_iterations):
        grad = derivative(x)
        x = x - learning_rate * grad  
        x_values.append(x)

        
        print(f"Iteration {i + 1}: x = {x:.5f}, y = {function(x):.5f}")

    return x, x_values


starting_x = 2  
learning_rate = 0.1  
num_iterations = 20  


min_x, x_values = gradient_descent(starting_x, learning_rate, num_iterations)

print(f"\nLocal minima occurs at x = {min_x:.5f}, y = {function(min_x):.5f}")


x_range = np.linspace(-10, 4, 100)
y_range = function(x_range)

plt.plot(x_range, y_range, label='y = (x + 3)^2')
plt.scatter(x_values, function(np.array(x_values)), color='red', label='Gradient Descent Steps')
plt.title("Gradient Descent to Find Local Minima")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
