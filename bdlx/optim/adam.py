"""
Adam: A Method for Stochastic Optimization
https://arxiv.org/abs/1412.6980
"""
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from bdlx.typing import Pytree


Param = Pytree
Batch = Pytree


class AdamState(NamedTuple):
    """State including position and momentums."""
    step: int
    position: Pytree
    momentum_mu: Pytree
    momentum_nu: Pytree


def step( # pylint: disable=too-many-arguments,too-many-locals
        state: AdamState,
        batch: Batch,
        loss_fn: Callable[[Param, Batch], Any],
        learning_rate: float,
        l2_regularizer: float = 0.0,
        wd_regularizer: float = 0.0,
        momentums: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        has_aux: bool = False,
        axis_name: Optional[str] = None,
        grad_mask: Optional[Callable[[Param], Param]] = None,
    ) -> Tuple[Any, AdamState]:
    """Updates state.

    Args:
        state: Current state.
        batch: It will be send to `loss_fn`.
        loss_fn: Loss function to be differentiated.
        learning_rate: Learning rate.
        l2_regularizer: L2 regularization coefficient.
        wd_regularizer: Decoupled weight decay regularization coefficient.
        momentums: Momentum coefficients.
        eps: Small value added to the denominator.
        has_aux: It indicates whether the `energy_fn` returns a pair, with the
            first element as the main output of the energy function for
            differentiation and the second element as optional auxiliary data.
        axis_name: `gradients` will be averaged across replicas if specified.
        grad_mask: It applies arbitrary transformation to `gradient`.

    Returns:
        Auxiliary data and updated state.
    """
    aux, gradient = jax.value_and_grad(
        loss_fn, argnums=0, has_aux=has_aux)(state.position, batch)
    if l2_regularizer > 0:
        gradient = jax.tree_util.tree_map(
            lambda p, g: p * l2_regularizer + g,
            state.position, gradient)
    if axis_name is not None:
        gradient = jax.lax.pmean(gradient, axis_name)
    if grad_mask is not None:
        gradient = grad_mask(gradient)

    momentum_mu = jax.tree_util.tree_map(
        lambda mu, g: momentums[0] * mu + (1.0 - momentums[0]) * g,
        state.momentum_mu, gradient)
    momentum_nu = jax.tree_util.tree_map(
        lambda nu, g: momentums[1] * nu + (1.0 - momentums[1]) * jnp.square(g),
        state.momentum_nu, gradient)

    momentum_mu_hat = jax.tree_util.tree_map(
        lambda mu: mu / (1.0 - momentums[0]**(state.step + 1)),
        momentum_mu)
    momentum_nu_hat = jax.tree_util.tree_map(
        lambda nu: nu / (1.0 - momentums[1]**(state.step + 1)),
        momentum_nu)
    position = jax.tree_util.tree_map(
        lambda p, mu, nu: \
            (1.0 - learning_rate * wd_regularizer) * p \
            - learning_rate * mu / (jnp.sqrt(nu) + eps),
        state.position, momentum_mu_hat, momentum_nu_hat)

    return aux, AdamState(
        step=state.step+1, position=position,
        momentum_mu=momentum_mu, momentum_nu=momentum_nu)
