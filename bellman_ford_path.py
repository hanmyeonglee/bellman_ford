import cupy as cp

Inf = 2 ** 30 - 1


def bellman_ford(edges, n_vertices, start):
    distance = cp.asarray([Inf] * n_vertices, dtype=cp.int32)
    previous = cp.asarray([Inf] * n_vertices, dtype=cp.int32)
    distance[start] = 0
    indice = cp.arange(n_vertices)

    for _ in range(n_vertices):
        temp_distance = distance + edges
        u = cp.argmin(temp_distance, axis=-1)
        previous = u
        distance = temp_distance[(indice, u)]

    path_info = previous.get()
    def get_path(destination):
        path = []
        while destination != start:
            path.append(destination)
            destination = path_info[destination]
        
        path.append(start)
        path.reverse()
        return path
    
    return distance, get_path


def make_path_intuitively(path):
    return '[' + ' -> '.join(map(str, path)) + ']'


edges = [
    [0, Inf, Inf, Inf],
    [5, 0, 2, Inf],
    [3, Inf, 0, Inf],
    [Inf, 6, 4, 0]
]
n_vertices = 4

edges = cp.asarray(edges, dtype=cp.int32)

distance, get_path = bellman_ford(edges, n_vertices, 0)

print(distance)

for node in range(n_vertices):
    print(f'{0} -> {node} : {make_path_intuitively(get_path(node))} (cost: {distance[node]})')