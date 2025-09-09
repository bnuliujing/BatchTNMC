import math

import numpy as np
import scipy
import torch


def _safe_svd_gpu(tensor, eps=1e-10):
    try:
        return torch.linalg.svd(tensor, full_matrices=False)
        # return torch.linalg.svd(tensor, full_matrices=False, driver="gesvd")  # no parallel
    except Exception as e:
        print(f"Error: {e}")
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            tensor = torch.where(
                torch.isnan(tensor) | torch.isinf(tensor),
                torch.tensor(eps, dtype=tensor.dtype, device=tensor.device),
                tensor,
            )
        padding = eps * torch.eye(tensor.shape[0], tensor.shape[1], dtype=tensor.dtype, device=tensor.device)
        tensor = tensor + padding
        return torch.linalg.svd(tensor, full_matrices=False)
        # return torch.linalg.svd(tensor, full_matrices=False, driver="gesvd")  # no parallel


def _safe_svd_cpu(tensor, eps=1e-10):
    try:
        return scipy.linalg.svd(tensor, full_matrices=False, lapack_driver="gesvd")  # numerically stable
    except Exception as e:
        print(f"Error: {e}")
        if np.isnan(tensor).any() or np.isinf(tensor).any():
            tensor = np.where(
                np.isnan(tensor) | np.isinf(tensor),
                eps,
                tensor,
            )
        padding = eps * np.eye(tensor.shape[0], tensor.shape[1])
        tensor = tensor + padding
        return scipy.linalg.svd(tensor, full_matrices=False, lapack_driver="gesvd")


def svd(tensor, chi):
    assert tensor.dim() == 2
    if tensor.device.type == "cuda":
        # svd on GPU
        U, S, Vh = _safe_svd_gpu(tensor)
        U = U[:, :chi]
        S = S[:chi]
        Vh = Vh[:chi, :]
    else:
        # svd on CPU
        U, S, Vh = _safe_svd_cpu(tensor.numpy())
        U = torch.from_numpy(U[:, :chi])
        S = torch.from_numpy(S[:chi])
        Vh = torch.from_numpy(Vh[:chi, :])

    return U, S, Vh


def batch_svd(tensor, chi):
    assert tensor.dim() == 3
    if tensor.device.type == "cuda":
        # svd on GPU in parallel
        U, S, Vh = _safe_svd_gpu(tensor)
        U = U[:, :, :chi]
        S = S[:, :chi]
        Vh = Vh[:, :chi, :]
    else:
        # svd on CPU sequentially
        U_list, S_list, Vh_list = [], [], []
        for i in range(tensor.shape[0]):
            U, S, Vh = _safe_svd_cpu(tensor[i].numpy())
            U_list.append(U)
            S_list.append(S)
            Vh_list.append(Vh)
        U, S, Vh = np.array(U_list), np.array(S_list), np.array(Vh_list)

        U = torch.from_numpy(U[:, :, :chi])
        S = torch.from_numpy(S[:, :chi])
        Vh = torch.from_numpy(Vh[:, :chi, :])

    return U, S, Vh


def graph_to_coupling(G, L):
    """
    Convert a networkx graph to a coupling matrix.
    The coupling matrix is a 3D tensor of shape (L, L, 4) where
    the last dimension represents the coupling to left, up, right, and down.
    """
    coupling_mat = torch.zeros((L, L, 4), dtype=torch.float64)
    for i in range(L):
        for j in range(L):
            if (i, j - 1) in G:
                coupling_mat[i, j, 0] = G[(i, j)][(i, j - 1)]["weight"]
            if (i - 1, j) in G:
                coupling_mat[i, j, 1] = G[(i, j)][(i - 1, j)]["weight"]
            if (i, j + 1) in G:
                coupling_mat[i, j, 2] = G[(i, j)][(i, j + 1)]["weight"]
            if (i + 1, j) in G:
                coupling_mat[i, j, 3] = G[(i, j)][(i + 1, j)]["weight"]

    return coupling_mat


def batch_graph_to_coupling(G, n_ins, L):
    """
    Convert a networkx graph to a coupling matrix.
    The coupling matrix is a 4D tensor of shape (n_ins, L, L, 4) where
    the last dimension represents the coupling to left, up, right, and down.
    """
    coupling_mat = torch.zeros((n_ins, L, L, 4), dtype=torch.float64)
    for i in range(L):
        for j in range(L):
            if (i, j - 1) in G:
                coupling_mat[:, i, j, 0] = torch.from_numpy(G[(i, j)][(i, j - 1)]["weight"])
            if (i - 1, j) in G:
                coupling_mat[:, i, j, 1] = torch.from_numpy(G[(i, j)][(i - 1, j)]["weight"])
            if (i, j + 1) in G:
                coupling_mat[:, i, j, 2] = torch.from_numpy(G[(i, j)][(i, j + 1)]["weight"])
            if (i + 1, j) in G:
                coupling_mat[:, i, j, 3] = torch.from_numpy(G[(i, j)][(i + 1, j)]["weight"])

    return coupling_mat


def multiply(mpo, mps):
    """
        b
        |
    a---O---c
        |
        d
        |
    i---S---j
    """
    assert len(mps) == len(mpo)
    for i in range(len(mps)):
        l1 = mpo[i].shape[0]
        l2 = mps[i].shape[0]
        d_phys = mps[i].shape[1]
        mps[i] = torch.einsum("abcd,idj->aibcj", mpo[i], mps[i]).reshape(l1 * l2, d_phys, -1)

    return mps


def eat(mpo_tensor, mps_tensor):
    """
        b
        |
    a---O---c
        |
        d
        |
    i---S---j
    A single MPO tensor eats a single MPS tensor.
    """
    l1 = mpo_tensor.shape[0]
    l2 = mps_tensor.shape[0]
    up = mpo_tensor.shape[1]
    mps_tensor = torch.einsum("abcd,idj->aibcj", mpo_tensor, mps_tensor).reshape(l1 * l2, up, -1)

    return mps_tensor


def batch_multiply(mpo, mps):
    assert len(mps) == len(mpo)
    for i in range(len(mps)):
        batch_dim = mps[i].shape[0]
        l1 = mpo[i].shape[1]
        l2 = mps[i].shape[1]
        d_phys = mps[i].shape[2]
        mps[i] = torch.einsum("zabcd,zidj->zaibcj", mpo[i], mps[i]).reshape(batch_dim, l1 * l2, d_phys, -1)

    return mps


def compress(mps, chi=None):
    # svd from left to right
    res = 0
    for i in range(len(mps) - 1):
        l = mps[i].shape[0]
        d_phys = mps[i].shape[1]
        r = mps[i + 1].shape[-1]
        tensor = torch.einsum("ijk,klm->ijlm", mps[i], mps[i + 1]).reshape(l * d_phys, r * d_phys)
        U, S, Vh = svd(tensor, chi=chi)
        mps[i] = U.reshape(l, d_phys, -1)
        mps[i + 1] = (torch.diag(S) @ Vh).reshape(-1, d_phys, r)
        norm = torch.norm(mps[i + 1])
        res += torch.log(norm)
        mps[i + 1] /= norm
    # svd from right to left
    for i in range(len(mps) - 1, 0, -1):
        l = mps[i - 1].shape[0]
        d_phys = mps[i].shape[1]
        r = mps[i].shape[-1]
        tensor = torch.einsum("ijk,klm->ijlm", mps[i - 1], mps[i]).reshape(l * d_phys, r * d_phys)
        U, S, Vh = svd(tensor, chi=chi)
        mps[i] = Vh.reshape(-1, d_phys, r)
        mps[i - 1] = (U @ torch.diag(S)).reshape(l, d_phys, -1)

    return res.item(), mps


def batch_compress(mps, chi=None):
    # svd from left to right
    res = 0
    for i in range(len(mps) - 1):
        batch_dim = mps[i].shape[0]
        l = mps[i].shape[1]
        d_phys = mps[i].shape[2]
        r = mps[i + 1].shape[-1]
        tensor = torch.einsum("zijk,zklm->zijlm", mps[i], mps[i + 1]).reshape(batch_dim, l * d_phys, r * d_phys)
        U, S, Vh = batch_svd(tensor, chi=chi)
        mps[i] = U.reshape(batch_dim, l, d_phys, -1)
        mps[i + 1] = torch.einsum("zb,zbc->zbc", S, Vh).reshape(batch_dim, -1, d_phys, r)
        norm = torch.norm(mps[i + 1].reshape(batch_dim, -1), dim=1)
        res += torch.log(norm)
        mps[i + 1] /= norm[:, None, None, None]
    # svd from right to left
    for i in range(len(mps) - 1, 0, -1):
        batch_dim = mps[i].shape[0]
        l = mps[i - 1].shape[1]
        d_phys = mps[i].shape[2]
        r = mps[i].shape[-1]
        tensor = torch.einsum("zijk,zklm->zijlm", mps[i - 1], mps[i]).reshape(batch_dim, l * d_phys, r * d_phys)
        U, S, Vh = batch_svd(tensor, chi=chi)
        mps[i] = Vh.reshape(batch_dim, -1, d_phys, r)
        mps[i - 1] = torch.einsum("zab,zb->zab", U, S).reshape(batch_dim, l, d_phys, -1)

    return res, mps


def compute_energy(s, Jij):
    """
    Computes the energy for a batch of spin configurations.
    Args:
        s: Tensor of shape (batch_size, L, L) containing spin configurations (±1).
        Jij: Tensor of shape (L, L, 4) containing couplings.
    Returns:
        energy: Tensor of shape (batch_size,) containing energies.
    """
    assert s.dim() == 3 and Jij.dim() == 3
    L = Jij.shape[0]
    energy = torch.zeros(s.size(0), device=Jij.device, dtype=Jij.dtype)
    # Horizontal interactions (right direction)
    energy += -torch.sum(Jij[:, : L - 1, 2] * s[:, :, : L - 1] * s[:, :, 1:], dim=(1, 2))
    # Vertical interactions (down direction)
    energy += -torch.sum(Jij[: L - 1, :, 3] * s[:, : L - 1, :] * s[:, 1:, :], dim=(1, 2))

    return energy


def metropolis(logq, energy, beta):
    """
    Perform Metropolis-Hastings sampling.
    Args:
        logq: Array of shape (N_s,) containing log probabilities.
        energy: Array of shape (N_s,) containing energies.
        beta: Inverse temperature.
    Returns:
        acc_rate: Acceptance rate.
        acc_list: List of accepted configuration indices.
    """
    num_accepted = 0
    num_transitions = logq.shape[0] - 1
    current_cfg_idx = 0
    acc_list = [current_cfg_idx]
    for i in range(1, num_transitions + 1):
        logq_last = logq[current_cfg_idx]
        energy_last = energy[current_cfg_idx]
        logq_new = logq[i]
        energy_new = energy[i]
        logp_accept = logq_last - logq_new - beta * energy_new + beta * energy_last
        if logp_accept >= 0:
            num_accepted += 1
            current_cfg_idx = i
        else:
            u = np.random.uniform(0, 1)
            if u < np.exp(logp_accept):
                num_accepted += 1
                current_cfg_idx = i
        acc_list.append(current_cfg_idx)
    acc_rate = num_accepted / num_transitions

    return acc_rate, acc_list


def set_ising_bmat(beta, Jij, dtype=torch.float64, device="cpu"):
    return torch.exp(beta * torch.tensor([[Jij, -Jij], [-Jij, Jij]], dtype=dtype, device=device))


def batch_set_ising_bmat(beta, Jij, dtype=torch.float64, device="cpu"):
    # Jij is a 1D tensor of shape (n_dis,)
    Jij = Jij[:, None, None]  # Expand to (n_dis, 1, 1)
    return torch.exp(beta * Jij * torch.tensor([[1, -1], [-1, 1]], dtype=dtype, device=device))
