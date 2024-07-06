import numpy as np
import torch

from .utils import create_dot_dict
from collections import defaultdict


def knn_graph_from_points(points, k):
    ## move to inside the func so the code can run without installing these packages
    from torch_geometric.nn import knn_graph
    from torch_sparse import coalesce

    edge_index = knn_graph(torch.FloatTensor(np.array(points)), k=k, loop=True)
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
    edge_index_torch, _ = coalesce(edge_index, None, len(points), len(points))
    edge_index = edge_index_torch.data.cpu().numpy().T
    edge_cost = defaultdict(list)
    edges = defaultdict(list)
    for i, edge in enumerate(edge_index):
        edge_cost[edge[1]].append(np.linalg.norm(points[edge[1]] - points[edge[0]]))
        edges[edge[1]].append(edge[0])

    return create_dot_dict(points=points, edges=edges, edge_index=edge_index, edge_cost=edge_cost)