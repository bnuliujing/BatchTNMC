"""Sampling class for 2D Ising spin glasses.
run:
python sampler.py --L 16 --chi 8 --beta 1 --p 0 --seed 1
"""

import math
import os

import fire
import torch
from opt_einsum import contract
from torch.distributions import Bernoulli
from tqdm import tqdm

from create_cache import create_cache_ising
from create_instances import create_rbim
from utils import compute_energy, metropolis, set_ising_bmat


class SamplerIsing:
    def __init__(self, L, chi, beta, p, seed, device="cpu"):
        self.L = L
        self.chi = chi
        self.beta = beta
        self.p = p
        self.seed = seed
        self.device = device

        ham_path = f"./instances/L{L}_p{p:.6f}_seed{seed}.pt"
        if not os.path.isfile(ham_path):
            raise FileNotFoundError(f"Hamiltonian file not found: {ham_path}")
        self.J_mat = torch.load(ham_path, weights_only=True, map_location=self.device)

    def sample(self, bs=1):
        cfgs = torch.zeros((bs, self.L, self.L), dtype=torch.bool, device=self.device)
        log_probs = torch.zeros(bs, dtype=torch.float, device=self.device)

        # draw samples in raster scan order
        for row in tqdm(range(self.L), desc="Sampling row"):
            # load mps cache, cfg_up, and J_mat_row into specified device
            cache_path = (
                f"./cache/L{self.L}_chi{self.chi}_beta{self.beta:.6f}_p{self.p:.6f}_seed{self.seed}/row{row}.pt"
            )
            if not os.path.isfile(cache_path):
                raise FileNotFoundError(f"MPS cache file not found: {cache_path}")
            mps = torch.load(cache_path, map_location=self.device, weights_only=True)  # mps[i]: (D, 2, D)
            cfg_up = cfgs[:, row - 1, :]  # (bs, L)
            J_mat_row = self.J_mat[row, :, :]  # (L, 4)

            row_cfgs, row_log_probs = self._sample_row(bs, mps, cfg_up, J_mat_row)
            cfgs[:, row, :] = row_cfgs
            log_probs += row_log_probs

        return cfgs, log_probs

    def _sample_row(self, bs, mps, cfg_up, J_mat_row):
        row_cfgs = torch.zeros((bs, self.L), dtype=torch.float, device=self.device)
        row_log_probs = torch.zeros((bs, self.L), dtype=torch.float, device=self.device)

        # build right environment given the configurations of the upper row and Boltzmann weights
        right_envs = {}  # right_envs[j] is the right env tensor for sampling the j-th column, shape (bs, D)
        right_envs[self.L - 1] = torch.ones(bs, 1, dtype=torch.float, device=self.device)
        for col in range(self.L - 2, -1, -1):
            B_mat = set_ising_bmat(self.beta, J_mat_row[col + 1, 1], dtype=torch.float, device=self.device)
            tmp = torch.einsum("ijk,lj->ilk", mps[col + 1], B_mat)
            tmp = tmp[:, cfg_up[:, col + 1].long(), :]  # (D_l, bs, D_r)
            right_envs[col] = torch.einsum("ibj,bj->bi", tmp, right_envs[col + 1])  # update right env
            right_envs[col] /= torch.norm(right_envs[col], dim=1, keepdim=True)

        left_env = torch.ones((bs, 1), dtype=torch.float, device=self.device)
        for col in range(self.L):
            rho = contract("bi,ijk,bk->bj", left_env, mps[col], right_envs[col]).abs()  # (bs, 2)
            up_field = 2 * self.beta * J_mat_row[col, 1] * (2 * cfg_up[:, col] - 1)
            probs = torch.sigmoid(torch.log(rho[:, 1] / rho[:, 0]) + up_field)
            dist = Bernoulli(probs=probs)
            row_cfgs[:, col] = dist.sample()
            row_log_probs[:, col] = dist.log_prob(row_cfgs[:, col])
            left_env = torch.einsum("bi,ibj->bj", left_env, mps[col][:, row_cfgs[:, col].long(), :])  # update left env
            left_env /= torch.norm(left_env, dim=1, keepdim=True)

        return row_cfgs, row_log_probs.sum(dim=1)


def test(L, chi, beta, p, seed, bs=1000, device="cuda:0"):
    # create Ising instance
    G = create_rbim(L, p, seed)

    # create mps cache
    create_cache_ising(L, chi, beta, p, seed, device=device)

    # draw samples
    sampler = SamplerIsing(L, chi, beta, p, seed, device=device)
    cfgs, log_probs = sampler.sample(bs=bs)
    energies = compute_energy(2 * cfgs - 1, sampler.J_mat)

    acc_rate, acc_list = metropolis(log_probs.cpu(), energies.cpu(), beta)
    print(f"\nMetropolis acceptance rate: {acc_rate:.4f}")


if __name__ == "__main__":
    fire.Fire(test)
