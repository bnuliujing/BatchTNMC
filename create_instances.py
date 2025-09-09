"""Create an instance for 2D spin glasses with open boundary conditions"""

import os

import fire
import networkx as nx
import numpy as np
import torch

from utils import graph_to_coupling


def create_rbim(L, p, seed=1):
    """
    Create an instance for 2D random-bond Ising model with open boundary conditions.

    Args:
        L (int): Lattice size.
        p (float): Probability of -1 bond, p=0 for Ising, p=0.5 for EA model.
        seed (int, optional): Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    graph = nx.grid_2d_graph(L, L)
    for u, v in graph.edges():
        graph[u][v]["weight"] = rng.choice([-1.0, 1.0], p=[p, 1 - p])
    coupling_mat = graph_to_coupling(graph, L)
    filename = f"./instances/L{L}_p{p:.6f}_seed{seed}.pt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(coupling_mat.float(), filename)
    print(f"Successfully created instance and saved to {filename}")

    return graph


if __name__ == "__main__":
    fire.Fire(create_rbim)
