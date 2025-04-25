# KrylobSubspaceLibrary/utils/hvp.py
# Hessian-vector product (HVP) for a given function
import torch


def hvp(loss_fn, model, inputs, targets, v):
    """
    Compute the Hessian-vector product of a given loss function with respect to the model parameters.

    Author: Austin M. McBurney
    Date: 2025-4-24
    Args:
        loss_fn: The loss function to compute the Hessian-vector product.
        model: The model for which the Hessian-vector product is computed.
        inputs: The input data.
        targets: The target data.
        v: The vector with respect to which the Hessian-vector product is computed.

    Returns:
        The Hessian-vector product.
    """
    # First forward and backward pass
    loss = loss_fn(model(inputs), targets)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    flat_grads = torch.cat([g.contiguous().view(-1) for g in grads])

    # Dot with vector
    dot = torch.dot(flat_grads, v)

    # Second backward pass
    Hv = torch.autograd.grad(dot, model.parameters(), retain_graph=True)
    flat_Hv = torch.cat([g.contiguous().view(-1) for h in Hv])

    return flat_Hv