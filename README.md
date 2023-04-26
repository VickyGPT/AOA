## selection sort

```
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_index = i
        for j in range(i+1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr

arr = []
size = int(input("Enter size of array: "))
for i in range(size):
    arr.append(int(input("Enter element {}: ".format(i+1))))

sorted_arr = selection_sort(arr)
print("Sorted array:", sorted_arr)

```
 
 
## merge sort

```
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]

    left = merge_sort(left)
    right = merge_sort(right)

    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result += left[i:]
    result += right[j:]

    return result

arr = []
size = int(input("Enter size of array: "))
for i in range(size):
    arr.append(int(input("Enter element {}: ".format(i+1))))

sorted_arr = merge_sort(arr)
print("Sorted array:", sorted_arr)


```

 
##  To implement fractional knapsack problem using Greedy method.

```
def fractional_knapsack(items, capacity):
    """
    Solve the fractional knapsack problem using the greedy method.

    Parameters:
    items - A list of tuples representing the items, where each tuple is of the form (weight, value).
    capacity - The maximum weight the knapsack can hold.

    Returns:
    The maximum value that can be obtained by filling the knapsack with the given capacity.
    """

    # Sort items in decreasing order of value per unit weight
    items = sorted(items, key=lambda x: x[1] / x[0], reverse=True)

    # Initialize variables
    total_value = 0
    total_weight = 0

    # Iterate through each item and add to knapsack until capacity is reached
    for item in items:
        weight = item[0]
        value = item[1]
        if total_weight + weight <= capacity:
            total_weight += weight
            total_value += value
        else:
            remaining_capacity = capacity - total_weight
            fraction = remaining_capacity / weight
            total_weight += remaining_capacity
            total_value += fraction * value
            break

    return total_value

# Take user input for items and capacity
items = []
n = int(input("Enter the number of items: "))
for i in range(n):
    weight = float(input("Enter the weight of item {}: ".format(i+1)))
    value = float(input("Enter the value of item {}: ".format(i+1)))
    items.append((weight, value))

capacity = float(input("Enter the capacity of the knapsack: "))

# Call the fractional_knapsack function and print the result
max_value = fractional_knapsack(items, capacity)
print("The maximum value that can be obtained is:", max_value)

```

 
##  

```

```

 
##

```

```

 
##

```

```

 
##

```

```

 
##

```

```

 
##

```

```

 
##

```

```

 
##

```

```

 
##

```

```

 
##

```

```

 
##

```

```

 
##

```

```

 
##

```

```

 
##

```

```

 
##

```

```

 
##

```

```

 
##

```

```

 
##

```

```

 
##

```

```

 
##

```

```

