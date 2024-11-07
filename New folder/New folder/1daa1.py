# Non-recursive function to calculate Fibonacci numbers
def fibonacci_iterative(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib_sequence = [0, 1]
    for i in range(2, n):
        next_value = fib_sequence[i - 1] + fib_sequence[i - 2]
        fib_sequence.append(next_value)
    
    return fib_sequence

# Example usage
n = int(input("Enter number of terms: "))
print("Fibonacci sequence (Iterative):", fibonacci_iterative(n))

# Recursive function to calculate Fibonacci numbers
def fibonacci_recursive(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib_sequence = fibonacci_recursive(n - 1)
    fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence

# Example usage
n = int(input("Enter number of terms: "))
print("Fibonacci sequence (Recursive):", fibonacci_recursive(n))
