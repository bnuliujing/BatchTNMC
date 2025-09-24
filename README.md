# BatchTNMC

Welcome to `BatchTNMC` [arXiv:2509.19006](https://arxiv.org/abs/2509.19006), a Python implementation for sampling 2D spin glasses using Tensor Network Monte Carlo (TNMC) [SciPost Phys. 14, 123 (2023)](https://scipost.org/10.21468/SciPostPhys.14.5.123), [Phys. Rev. B 111, 094201](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.111.094201) with GPU.
This repository provides tools for generating RBIM (Random Bond Ising Model) instances, creating tensor network caches, and performing efficient sampling using boundary Matrix Product States (bMPS) techniques.

## Requirements

* Python 3.10
* PyTorch 2.7.1
* NumPy 2.1.2
* SciPy 1.15.3
* NetworkX 3.3
* Fire 0.7.0
* tqdm 4.67.1
* opt_einsum 3.4.0

## Package Architecture

The package includes the following key components:

* `create_instances.py` - Functions for generating RBIM instances with specified lattice sizes and bond disorder parameters.
* `create_cache.py` - Functions for precomputing MPS caches needed for efficient sampling.
* `sampler.py` - Core sampling implementation using MPS tensor networks for efficient spin configuration generation.
* `utils.py` - Utility functions for tensor operations, SVD decompositions, energy computations, and other helper functions.
* `batch_*.py` - Batch processing versions for handling multiple instances simultaneously.

## Example

Our code can simulate system size $L = 1024$ with 1000 independent disorder realizations, and for each disorder realization drawing 1000 samples in parallel:

```
python batch_sampler.py --n_dis 1000 --L 1024 --chi 8 --bs 1000 --create_ins True --create_cache True --sampling True
```

This is only for demonstration purposes, since it requires at least 52GB of GPU memory, approximately 1TB of system RAM, and 500GB of available disk space, with an estimated runtime of around 3 hours and 20 minutes on an A100 GPU.

See also `tutorial.ipynb` for a simple example on 2D Ising model.

## Citation

If you use this code, please cite our paper:

**arXiv**: https://arxiv.org/abs/2509.19006

```bibtex
@misc{chen2025batchtnmc,
      title={BatchTNMC: Efficient sampling of two-dimensional spin glasses using tensor network Monte Carlo}, 
      author={Tao Chen and Jingtong Zhang and Jing Liu and Youjin Deng and Pan Zhang},
      year={2025},
      eprint={2509.19006},
      archivePrefix={arXiv},
      primaryClass={cond-mat.stat-mech},
      url={https://arxiv.org/abs/2509.19006}, 
}
```