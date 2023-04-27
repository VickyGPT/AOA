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

 
##   minimum cost spanning tree using Kruskal

```
# Kruskal's algorithm implementation for minimum cost spanning tree
from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        self.V = vertices # Number of vertices
        self.graph = [] # Graph with edges and their weights
    
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])
    
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])
    
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1
    
    def kruskalMST(self):
        result = [] # Minimum Spanning Tree
        i = 0 # Index variable, used for sorted edges
        e = 0 # Index variable, used for result[]
        
        self.graph = sorted(self.graph, key=lambda item: item[2])
        
        parent = []
        rank = []
        
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
        
        while e < self.V - 1:
            u, v, w = self.graph[i]
            i += 1
            x = self.find(parent, u)
            y = self.find(parent, v)
            
            if x != y:
                e += 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
        
        return result
    
# Driver code
V = int(input("Enter the number of vertices: "))
g = Graph(V)

E = int(input("Enter the number of edges: "))
for i in range(E):
    print("Enter the source, destination, and weight of edge", i+1)
    u, v, w = map(int, input().split())
    g.addEdge(u, v, w)

# Construct and print Minimum Spanning Tree
print("Minimum Spanning Tree using Kruskal's algorithm:")
mst = g.kruskalMST()
for u, v, weight in mst:
    print(f"{u} - {v}: {weight}")

```

 
## m s p prims

```

# Prim's algorithm implementation for minimum cost spanning tree
import sys

class Graph:
    def __init__(self, vertices):
        self.V = vertices # Number of vertices
        self.graph = [[0 for column in range(vertices)] for row in range(vertices)] # Graph with edges and their weights
    
    def printMST(self, parent):
        print("Minimum Spanning Tree using Prim's algorithm:")
        for i in range(1, self.V):
            print(f"{parent[i]} - {i}: {self.graph[i][parent[i]]}")
    
    def minKey(self, key, mstSet):
        min_value = sys.maxsize
        min_index = -1
        
        for v in range(self.V):
            if key[v] < min_value and mstSet[v] == False:
                min_value = key[v]
                min_index = v
        
        return min_index
    
    def primMST(self):
        key = [sys.maxsize] * self.V # Key values used to pick minimum weight edge in cut
        parent = [None] * self.V # Array to store constructed MST
        key[0] = 0 # Make key 0 to pick the first vertex
        mstSet = [False] * self.V # Set all vertices as not yet included in MST
        
        parent[0] = -1 # First node is always the root of MST
        
        for cout in range(self.V):
            u = self.minKey(key, mstSet)
            mstSet[u] = True
            
            for v in range(self.V):
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u
        
        self.printMST(parent)

# Driver code
V = int(input("Enter the number of vertices: "))
g = Graph(V)

for i in range(V):
    print(f"Enter the weights of edges for vertex {i}")
    weights = list(map(int, input().split()))
    for j in range(V):
        g.graph[i][j] = weights[j]

# Construct and print Minimum Spanning Tree
g.primMST()


```

 
## All Pairs Shortest Path algorithm using Dynamic programming

```
INF = float('inf')

# Function to compute the shortest path between all pairs of vertices
def floyd_warshall(graph):
    n = len(graph)
    dist = [[0 if i == j else graph[i][j] if graph[i][j] != 0 else INF for j in range(n)] for i in range(n)]
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist

# Taking user input for the graph
n = int(input("Enter the number of vertices: "))
graph = []
print("Enter the adjacency matrix of the graph:")
for i in range(n):
    row = list(map(int, input().split()))
    graph.append(row)

# Computing and printing the shortest path between all pairs of vertices
dist = floyd_warshall(graph)
print("The minimum distances between all pairs of vertices are:")
for i in range(n):
    for j in range(n):
        if dist[i][j] == INF:
            print("INF", end=" ")
        else:
            print(dist[i][j], end=" ")
    print()

```

 
##  Longest common subsequence algorithm

```
def lcs(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    lcs_len = dp[m][n]
    lcs_seq = ""
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            lcs_seq = seq1[i - 1] + lcs_seq
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    return lcs_len, lcs_seq

# Taking user input for the sequences
seq1 = input("Enter the first sequence: ")
seq2 = input("Enter the second sequence: ")

# Computing and printing the length and sequence of the LCS
lcs_len, lcs_seq = lcs(seq1, seq2)
print("The length of the longest common subsequence is:", lcs_len)
print("The longest common subsequence is:", lcs_seq)

```

 
## n queen

```
 def is_safe(board, row, col, n):
    # Check row on left side
    for i in range(col):
        if board[row][i] == 1:
            return False

    # Check upper diagonal on left side
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Check lower diagonal on left side
    for i, j in zip(range(row, n), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Queen can be placed in this position
    return True


def solve_n_queens(board, col, n):
    # Base case: all queens are placed
    if col >= n:
        return True

    # Try placing this queen in all rows one by one
    for i in range(n):
        if is_safe(board, i, col, n):
            board[i][col] = 1

            # Recur to place rest of the queens
            if solve_n_queens(board, col + 1, n):
                return True

            # If placing queen in board[i][col] doesn't lead to a solution,
            # then remove queen from board[i][col]
            board[i][col] = 0

    # If queen can't be placed in any row in this column, then return False
    return False


def print_board(board):
    for row in board:
        print(" ".join(str(cell) for cell in row))


n = int(input("Enter value of N: "))
board = [[0 for _ in range(n)] for _ in range(n)]

if solve_n_queens(board, 0, n):
    print("Solution found:")
    print_board(board)
else:
    print("No solution found")

```

 
## Naive String 

```
def naive_string_matching(text, pattern):
    m = len(pattern)
    n = len(text)
    for i in range(n-m+1):
        j = 0
        while j < m and text[i+j] == pattern[j]:
            j += 1
        if j == m:
            print("Pattern found at index", i)
            
text = "ABAAABCDBBABCDDEBCABC"
pattern = "ABC"
naive_string_matching(text, pattern)


```

 
## Graph coloring

```
class GraphColoring:
    def __init__(self):
        self.V = 4
        self.color = []

    def isSafeToColor(self, v, graphMatrix, color, c):
        # check for each edge
        for i in range(self.V):
            if graphMatrix[v][i] == 1 and c == color[i]:
                return False
        return True

    def graphColorUtil(self, graphMatrix, m, color, v):
        # If all vertices are assigned a color then return true
        if v == self.V:
            return True

        # Try different colors for vertex V
        for i in range(1, m+1):
            # check for assignment safety
            if self.isSafeToColor(v, graphMatrix, color, i):
                color[v] = i
                # recursion for checking other vertices
                if self.graphColorUtil(graphMatrix, m, color, v + 1):
                    return True
                # if color doesn't lead to solution
                color[v] = 0

        # If no color can be assigned to vertex
        return False

    def printColoringSolution(self, color):
        print("Color schema for vertices are:")
        for i in range(self.V):
            print(color[i])

    def graphColoring(self, graphMatrix, m):
        # Initialize all color values as 0.
        self.color = [0] * self.V

        # Call graphColorUtil() for vertex 0
        if not self.graphColorUtil(graphMatrix, m, self.color, 0):
            print("Color schema not possible")
            return False

        # Print the color schema of vertices
        self.printColoringSolution(self.color)
        return True

# Driver code
if __name__ == "__main__":
    graph_algo = GraphColoring()

    graphMatrix = [
        [0, 1, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 0],
    ]
    m = 3  # Number of colors
    graph_algo.graphColoring(graphMatrix, m)

```

 
## rabin karp

```

def rabin_karp(text, pattern):
    n = len(text)
    m = len(pattern)
    if n < m:
        return -1
    q = 101
    pattern_hash = sum(ord(pattern[i]) * pow(q, i) for i in range(m)) % q
    window_hash = sum(ord(text[i]) * pow(q, i) for i in range(m)) % q
    if pattern_hash == window_hash:
        if pattern == text[:m]:
            return 0
    for i in range(m, n):
        window_hash -= ord(text[i - m]) * pow(q, 0)
        window_hash *= q
        window_hash += ord(text[i])
        if pattern_hash == window_hash:
            if pattern == text[i - m + 1:i + 1]:
                return i - m + 1
    return -1

text = input("Enter the text: ")
pattern = input("Enter the pattern to search for: ")
index = rabin_karp(text, pattern)
if index == -1:
    print(f"The pattern '{pattern}' was not found in the text.")
else:
    print(f"The pattern '{pattern}' was found at index {index} in the text.")


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

