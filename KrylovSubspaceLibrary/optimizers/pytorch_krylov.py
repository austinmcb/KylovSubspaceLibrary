# KrylovSubspaceLibrary/optimizers/pytorch_krylov.py
import torch
from torch.optim.optimizer import Optimizer
from KrylovSubspaceLibrary.solvers.conjugate_gradient import conjugate_gradient
from KrylovSubspaceLibrary.utils.hvp import hvp