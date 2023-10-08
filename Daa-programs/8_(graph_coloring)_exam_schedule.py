import networkx as nx

def minimum_time_slots(G):
    chromatic_number = nx.coloring.greedy_color(G, strategy="largest_first")
    return max(chromatic_number.values()) + 1


subjects = ['Math', 'Physics', 'Chemistry', 'Biology']
students = {
'Math': ['Alice', 'Bob', 'Charlie'],
'Physics': ['Alice', 'Charlie', 'David'],
'Chemistry': ['Bob', 'Charlie', 'Eve'],
'Biology': ['Alice', 'David', 'Eve']
}
graph = nx.Graph()
graph.add_edge('Math', 'Physics')
graph.add_edge('Math', 'Chemistry')
graph.add_edge('Physics', 'Chemistry')
graph.add_edge('Physics', 'Biology')

min_time_slots = minimum_time_slots(G=graph)
print(f"Minimum time slots needed: {min_time_slots}")
