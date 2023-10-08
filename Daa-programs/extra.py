#3 dijstra
import itertools
import sys

def tsp(graph):
    num_nodes = len(graph)
    min_cost = sys.maxsize
    optimal_route = []

    for perm in itertools.permutations(range(num_nodes)):
        cost = sum(graph[perm[i - 1]][perm[i]] for i in range(num_nodes))
        if cost < min_cost:
            min_cost = cost
            optimal_route = perm

    return optimal_route, min_cost

# Example usage
graph = [
    [0, 10, 20, 30],
    [10, 0, 5, 15],
    [20, 5, 0, 8],
    [30, 15, 8, 0]
]

optimal_route, total_cost = tsp(graph)
print("Optimal Route:", optimal_route)
print("Total Cost:", total_cost)



#9 . queen 

def isSafe(mat, r, c):
for i in range(r):
if mat[i][c] == 'Q':
return False
if c - r + i >= 0 and mat[i][c - r + i] == 'Q':
return False
if c + r - i < len(mat) and mat[i][c + r - i] == 'Q':
return False
return True
def nQueen(mat, r):
if r == len(mat):
printSolution(mat)
return
for i in range(len(mat)):
if isSafe(mat, r, i):
mat[r][i] = 'Q'
nQueen(mat, r + 1)
mat[r][i] = '-'
def printSolution(mat):
for r in mat:
print(''.join(r))
print()
if _name_ == '_main_':
print("Enter the number of queens")
N = int(input())
mat = [['-' for _ in range(N)] for _ in range(N)]
nQueen(mat, 0)
