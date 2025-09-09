"""Create the MPS cache and save for sampling, batch version
mps cache is saved separately for each row
"""

import os

import fire
import torch
import tqdm

from utils import batch_compress, batch_multiply, batch_set_ising_bmat


def create_cache_ising(n_dis, L, chi, beta, p, seed, device="cpu"):
    # load Hamiltonian
    ham_path = f"./instances/n{n_dis}_L{L}_p{p:.6f}_seed{seed}.pt"
    if not os.path.isfile(ham_path):
        raise FileNotFoundError(f"Hamiltonian file not found: {ham_path}")
    J_mat = torch.load(ham_path, weights_only=True, map_location=device)  # (n_dis, L, L, 4)

    # define copy tensor for constructing the MPS and MPO
    I2 = torch.eye(2, device=device, dtype=torch.float64)
    I3 = torch.zeros((2, 2, 2), device=device, dtype=torch.float64)
    I4 = torch.zeros((2, 2, 2, 2), device=device, dtype=torch.float64)
    for i in range(2):
        I3[i, i, i] = 1
        I4[i, i, i, i] = 1

    # define the MPS and MPO from bottom row to top row
    # MPS is a list of tensors [(n_dis, 1, 2, D), (n_dis, D, 2, D), ..., (n_dis, D, 2, 1)]
    # MPO is a list of tensors [(n_dis, 1, 2, 2, 2), (n_dis, 2, 2, 2, 2), ..., (n_dis, 2, 2, 1, 2)]
    # order: lurd

    # create the MPS for the last row
    mps = []
    B_mat = batch_set_ising_bmat(beta, J_mat[:, -1, 0, 2], device=device)  # (n_dis, 2, 2)
    mps.append(torch.einsum("ij,zjk->zik", I2, B_mat).unsqueeze(1))  # (n_dis, 1, 2, D)
    for col in range(1, L - 1):
        B_mat = batch_set_ising_bmat(beta, J_mat[:, -1, col, 2], device=device)
        mps.append(torch.einsum("ijk,zkl->zijl", I3, B_mat))  # (n_dis, D, 2, D)
    mps.append(I2.unsqueeze(-1).repeat(n_dis, 1, 1, 1))  # (n_dis, D, 2, 1)
    cache_path = f"./cache/n{n_dis}_L{L}_chi{chi}_beta{beta:.6f}_p{p:.6f}_seed{seed}/row{L-1}.pt"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save([tensor.clone().cpu().float() for tensor in mps], cache_path)

    for row in tqdm.tqdm(range(L - 2, -1, -1), desc="Building MPS cache"):
        # create the MPO for current row
        mpo = []
        B_mat_d = batch_set_ising_bmat(beta, J_mat[:, row, 0, 3], device=device)
        B_mat_r = batch_set_ising_bmat(beta, J_mat[:, row, 0, 2], device=device)
        mpo.append((torch.einsum("ijk,zjl,zkm->zilm", I3, B_mat_r, B_mat_d).unsqueeze(1)))  # (n_dis, 1, 2, 2, 2)
        for col in range(1, L - 1):
            B_mat_d = batch_set_ising_bmat(beta, J_mat[:, row, col, 3], device=device)
            B_mat_r = batch_set_ising_bmat(beta, J_mat[:, row, col, 2], device=device)
            mpo.append(torch.einsum("ijkl,zkm,zln->zijmn", I4, B_mat_r, B_mat_d))  # (n_dis, 2, 2, 2, 2)
        B_mat_d = batch_set_ising_bmat(beta, J_mat[:, row, L - 1, 3], device=device)
        mpo.append((torch.einsum("ijk,zkl->zijl", I3, B_mat_d).unsqueeze(3)))  # (n_dis, 2, 2, 1, 2)

        # MPO x MPS, compress the new MPS, then save it
        mps = batch_multiply(mpo, mps)
        _, mps = batch_compress(mps, chi=chi)
        cache_path = f"./cache/n{n_dis}_L{L}_chi{chi}_beta{beta:.6f}_p{p:.6f}_seed{seed}/row{row}.pt"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save([tensor.clone().cpu().float() for tensor in mps], cache_path)
    print(
        f"Successfully created MPS cache and saved to ./cache/n{n_dis}_L{L}_chi{chi}_beta{beta:.6f}_p{p:.6f}_seed{seed}/"
    )


if __name__ == "__main__":
    fire.Fire(create_cache_ising)
