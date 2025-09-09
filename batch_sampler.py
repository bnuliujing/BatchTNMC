"""Sampling class for 2D Ising spin glasses, batch version
example:
python batch_sampler.py

for L=1024x1024, 1000 disorders, 1000 samples, run:
python batch_sampler.py --n_dis 1000 --L 1024 --chi 8 --bs 1000 --create_ins True --create_cache True --sampling True
"""

import math
import os

import fire
import torch
from torch.distributions import Bernoulli
from tqdm import tqdm

from batch_create_cache import create_cache_ising
from batch_create_instances import create_rbim
from utils import batch_set_ising_bmat, compute_energy, metropolis


class SamplerIsing:
    def __init__(self, n_dis, L, chi, beta, p, seed, device="cpu"):
        self.n_dis = n_dis
        self.L = L
        self.chi = chi
        self.beta = beta
        self.p = p
        self.seed = seed
        self.device = device

        ham_path = f"./instances/n{n_dis}_L{L}_p{p:.6f}_seed{seed}.pt"
        if not os.path.isfile(ham_path):
            raise FileNotFoundError(f"Hamiltonian file not found: {ham_path}")
        self.J_mat = torch.load(ham_path, weights_only=True, map_location="cpu")
        self._index = torch.arange(self.n_dis, device=self.device).unsqueeze(1)  # index helper

    def sample(self, bs=1):
        cfgs = torch.zeros((self.n_dis, bs, self.L, self.L), dtype=torch.bool, device="cpu")
        log_probs = torch.zeros((self.n_dis, bs), dtype=torch.float, device="cpu")

        # draw samples in zigzag order
        for row in tqdm(range(self.L), desc="Sampling row"):
            # load mps cache, cfg_up, and J_mat_row into specified device
            cache_path = f"./cache/n{self.n_dis}_L{self.L}_chi{self.chi}_beta{self.beta:.6f}_p{self.p:.6f}_seed{self.seed}/row{row}.pt"
            if not os.path.isfile(cache_path):
                raise FileNotFoundError(f"MPS cache file not found: {cache_path}")
            mps = torch.load(cache_path, map_location=self.device, weights_only=True)  # mps[i]: (n_dis, D, 2, D)
            cfg_up = cfgs[:, :, row - 1, :].to(self.device)  # (n_dis, bs, L)
            J_mat_row = self.J_mat[:, row, :, :].to(self.device)  # (n_dis, L, 4)

            row_cfgs, row_log_probs = self._sample_row(bs, mps, cfg_up, J_mat_row)
            cfgs[:, :, row, :] = row_cfgs.cpu()
            log_probs += row_log_probs.cpu()

        return cfgs, log_probs

    def _sample_row(self, bs, mps, cfg_up, J_mat_row):
        row_cfgs = torch.zeros((self.n_dis, bs, self.L), dtype=torch.float, device=self.device)
        row_log_probs = torch.zeros((self.n_dis, bs, self.L), dtype=torch.float, device=self.device)

        # build right environment given the configurations of the upper row and Boltzmann weights
        right_envs = {}  # right_envs[j] is the right env tensor for sampling the j-th column, shape (n_dis, bs, D)
        right_envs[self.L - 1] = torch.ones(self.n_dis, bs, 1, dtype=torch.float, device=self.device)
        for col in range(self.L - 2, -1, -1):
            B_mat = batch_set_ising_bmat(self.beta, J_mat_row[:, col + 1, 1], dtype=torch.float, device=self.device)
            tmp = torch.einsum("zijk,zlj->zilk", mps[col + 1], B_mat)  # (n_dis, D_l, 2, D_r)
            tmp = tmp[self._index, :, cfg_up[:, :, col + 1].long(), :]  # (n_dis, bs, D_l, D_r)
            right_envs[col] = torch.einsum("zbij,zbj->zbi", tmp, right_envs[col + 1])  # update right env
            right_envs[col] /= torch.norm(right_envs[col], dim=2, keepdim=True)

        left_env = torch.ones((self.n_dis, bs, 1), dtype=torch.float, device=self.device)
        for col in range(self.L):
            rho = torch.einsum("zbi,zijk,zbk->zbj", left_env, mps[col], right_envs[col]).abs()  # (n_dis, bs, 2)
            up_field = 2 * self.beta * J_mat_row[:, col, 1][:, None] * (2 * cfg_up[:, :, col] - 1)
            probs = torch.sigmoid(torch.log(rho[:, :, 1] / rho[:, :, 0]) + up_field)
            dist = Bernoulli(probs=probs)
            row_cfgs[:, :, col] = dist.sample()
            row_log_probs[:, :, col] = dist.log_prob(row_cfgs[:, :, col])
            tmp = mps[col][self._index, :, row_cfgs[:, :, col].long(), :]  # (n_dis, bs, D_l, D_r)
            left_env = torch.einsum("zbi,zbij->zbj", left_env, tmp)  # update left env
            left_env /= torch.norm(left_env, dim=2, keepdim=True)

        return row_cfgs, row_log_probs.sum(dim=2)


def test(
    n_dis=10,
    L=16,
    chi=8,
    beta=1.0,
    p=0.5,
    seed=1,
    bs=1000,
    device="cuda:0",
    create_ins=True,
    create_cache=True,
    sampling=True,
):
    # create Ising instance
    if create_ins:
        create_rbim(n_dis, L, p, seed)

    # create MPS cache
    if create_cache:
        create_cache_ising(n_dis, L, chi, beta, p, seed, device=device)

    # draw samples
    if sampling:
        sampler = SamplerIsing(n_dis, L, chi, beta, p, seed, device=device)
        cfgs, log_probs = sampler.sample(bs=bs)
        for idx in range(n_dis):
            energies = compute_energy(2.0 * cfgs[idx] - 1, sampler.J_mat[idx])
            acc_rate, acc_list = metropolis(log_probs[idx], energies, beta)
            print(f"\nMetropolis acceptance rate for disorder {idx}: {acc_rate:.4f}")
            # logw = -beta * energies - log_probs[idx]
            # logZ_hat = torch.logsumexp(logw, dim=0) - math.log(bs)
            # print(f"Estimated logZ: {logZ_hat.item()}")


if __name__ == "__main__":
    fire.Fire(test)
