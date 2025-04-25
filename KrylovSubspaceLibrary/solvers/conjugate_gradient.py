# KrylovSubspaceLibrary/solvers/conjugate_gradient.py
import torch

def conjugate_gradient(Av_fn, b, x0=None, tol=1e-5, max_iter=50):
    """
    Solves Ax = b using the conjugate gradient method, where A is not given explicitly,
      but as a function Ac_fn(v) = A @ v.
    
    Author: Austin M. McBurney
    Date: 2025-4-24
    Args:
        Av_fn (callable): Function that computes the matrix-vector product A @ v.
        b (Tensor): Right-hand side vector.
        x0 (Tensor): Initial guess for the solution (default is a zero vector).
        tol (float): Tolerance for convergence (default is 1e-5).
        max_iter (int): Maximum number of iterations (default is 50).
    
    Returns:
        x (Tensor): Approximate solution vector to Ax = b.
    """
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()
    
    r = b - Av_fn(x)
    p = r.clone()
    rs_old = torch.dot(r, r)

    for i in range(max_iter):
        Ap = Av_fn(p)
        alpha = rs_old / torch.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = torch.dot(r, r)

        if torch.sqrt(rs_new) < tol:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x