"""Create multiple instances for 2D spin glasses with open boundary conditions"""

import os

import fire
import networkx as nx
import numpy as np
import torch

from utils import batch_graph_to_coupling


def create_rbim(n_dis, L, p, seed=1):
    """
    Create multiple instances for 2D random-bond Ising model with open boundary conditions.

    Args:
        n_dis (int): Number of disorders.
        L (int): Lattice size.
        p (float): Probability of -1 bond.
        seed (int, optional): Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    graph = nx.grid_2d_graph(L, L)
    for u, v in graph.edges():
        graph[u][v]["weight"] = rng.choice([-1.0, 1.0], size=n_dis, p=[p, 1 - p])
    coupling_mat = batch_graph_to_coupling(graph, n_dis, L)
    filename = f"./instances/n{n_dis}_L{L}_p{p:.6f}_seed{seed}.pt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(coupling_mat.float(), filename)
    print(f"Successfully created {n_dis} instances and saved to {filename}")


if __name__ == "__main__":
    fire.Fire(create_rbim)
