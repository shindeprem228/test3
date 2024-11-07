def fractional_knapsack(value, weight, capacity):
    
    items = sorted([(v / w, w, v) for v, w in zip(value, weight)], reverse=True)
    
    max_value = 0   
    fractions = [0] * len(value)  

   
    for i, (ratio, w, v) in enumerate(items):
        if capacity > 0 and w <= capacity:
           
            fractions[i] = 1
            max_value += v
            capacity -= w
        else:
          
            fractions[i] = capacity / w if w > 0 else 0
            max_value += v * fractions[i]
            break  

    return max_value, fractions


n = int(input("Enter the number of items: "))
value = list(map(int, input("Enter the values of the items: ").split()))
weight = list(map(int, input("Enter the weights of the items: ").split()))
capacity = int(input("Enter the knapsack capacity: "))


max_value, fractions = fractional_knapsack(value, weight, capacity)


print("The maximum value of items that can be carried:", max_value)
print("The fractions of items to take:", fractions)
