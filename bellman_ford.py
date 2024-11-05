import cupy as cp

Inf = 2 ** 30 - 1


def bellman_ford(edges, n_vertices, start):
    distance = cp.asarray([Inf] * n_vertices, dtype=cp.int32)
    distance[start] = 0

    for _ in range(n_vertices):
        distance = cp.min(distance + edges, axis=-1)
    
    return distance

edges = [
    [0, Inf, Inf, Inf],
    [5, 0, 2, Inf],
    [3, Inf, 0, Inf],
    [Inf, 6, 4, 0]
]

edges = cp.asarray(edges, dtype=cp.int32)

print(bellman_ford(edges, 4, 0))