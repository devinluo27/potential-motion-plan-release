import numpy as np

INFINITY = np.inf

def min_dist(q, dist):
    """
    Returns the node with the smallest distance in q.
    """
    min_node = None
    for node in q:
        if min_node is None:
            min_node = node
        elif dist[node] < dist[min_node]:
            min_node = node

    return min_node


def dijkstra(nodes, edges, costs, source):
    """
    dijkstra search in configuration space without collision check
    """

    q = set()
    dist = {}

    for v in nodes:  # initialization
        dist[v] = INFINITY  # unknown distance from goal to v
        q.add(v)  # all nodes initially in q (unvisited nodes)

    # distance from goal to every node
    dist[source] = 0

    while q:
        # node with the least distance selected first
        u = min_dist(q, dist)

        q.remove(u)

        for index, v in enumerate(edges[u]):
            alt = dist[u] + costs[u][index]
            if alt < dist[v]:
                # a shorter path to v has been found
                dist[v] = alt

    return dist

