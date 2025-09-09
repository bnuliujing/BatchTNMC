"""Create the MPS cache and save for sampling.
mps cache is saved separately for each row
"""

import os

import fire
import torch
import tqdm

from utils import compress, multiply, set_ising_bmat


def create_cache_ising(L, chi, beta, p, seed, device="cpu"):
    # load Hamiltonian
    ham_path = f"./instances/L{L}_p{p:.6f}_seed{seed}.pt"
    if not os.path.isfile(ham_path):
        raise FileNotFoundError(f"Hamiltonian file not found: {ham_path}")
    J_mat = torch.load(ham_path, weights_only=True, map_location=device)  # (L, L, 4)

    # define copy tensor for constructing the MPS and MPO
    I2 = torch.eye(2, device=device, dtype=torch.float64)
    I3 = torch.zeros((2, 2, 2), device=device, dtype=torch.float64)
    I4 = torch.zeros((2, 2, 2, 2), device=device, dtype=torch.float64)
    for i in range(2):
        I3[i, i, i] = 1
        I4[i, i, i, i] = 1

    # define the MPS and MPO from bottom row to top row
    # MPS is a list of tensors [(1, 2, D), (D, 2, D), ..., (D, 2, 1)]
    # MPO is a list of tensors [(1, 2, 2, 2), (2, 2, 2, 2), ..., (2, 2, 1, 2)]
    # order: lurd

    mps = []
    B_mat = set_ising_bmat(beta, J_mat[-1, 0, 2], device=device)
    mps.append((I2 @ B_mat).unsqueeze(0))  # (1, 2, D)
    for col in range(1, L - 1):
        B_mat = set_ising_bmat(beta, J_mat[-1, col, 2], device=device)
        mps.append((I3 @ B_mat))  # (D, 2, D)
    mps.append(I2.unsqueeze(-1))  # (D, 2, 1)
    cache_path = f"./cache/L{L}_chi{chi}_beta{beta:.6f}_p{p:.6f}_seed{seed}/row{L-1}.pt"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save([tensor.clone().cpu().float() for tensor in mps], cache_path)

    for row in tqdm.tqdm(range(L - 2, -1, -1), desc="Building MPS cache"):
        # create the MPO for current row
        mpo = []
        B_mat_d = set_ising_bmat(beta, J_mat[row, 0, 3], device=device)
        B_mat_r = set_ising_bmat(beta, J_mat[row, 0, 2], device=device)
        mpo.append((torch.einsum("ijk,jl,km->ilm", I3, B_mat_r, B_mat_d).unsqueeze(0)))  # (1, 2, 2, 2)
        for col in range(1, L - 1):
            B_mat_d = set_ising_bmat(beta, J_mat[row, col, 3], device=device)
            B_mat_r = set_ising_bmat(beta, J_mat[row, col, 2], device=device)
            mpo.append(torch.einsum("ijkl,km,ln->ijmn", I4, B_mat_r, B_mat_d))  # (2, 2, 2, 2)
        B_mat_d = set_ising_bmat(beta, J_mat[row, L - 1, 3], device=device)
        mpo.append((torch.einsum("ijk,kl->ijl", I3, B_mat_d).unsqueeze(2)))  # (2, 2, 1, 2)

        # MPO x MPS, compress the new MPS, then save it
        mps = multiply(mpo, mps)
        _, mps = compress(mps, chi=chi)
        cache_path = f"./cache/L{L}_chi{chi}_beta{beta:.6f}_p{p:.6f}_seed{seed}/row{row}.pt"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save([tensor.clone().cpu().float() for tensor in mps], cache_path)
    print(f"Successfully created MPS cache and saved to ./cache/L{L}_chi{chi}_beta{beta:.6f}_p{p:.6f}_seed{seed}/")


if __name__ == "__main__":
    fire.Fire(create_cache_ising)
