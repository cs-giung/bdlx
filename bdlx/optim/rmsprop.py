"""
RMSProp: Divide the gradient by a running average of its recent magnitude:
- https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
- https://arxiv.org/abs/1308.0850
"""
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from bdlx.typing import Pytree


Param = Pytree
Batch = Pytree


class RMSPropState(NamedTuple):
    """State including position and momentums."""
    step: int
    position: Pytree
    momentum: Pytree
    momentum_mu: Pytree
    momentum_nu: Pytree


def step( # pylint: disable=too-many-arguments,too-many-locals
        state: RMSPropState,
        batch: Batch,
        loss_fn: Callable[[Param, Batch], Any],
        learning_rate: float,
        l2_regularizer: float = 0.0,
        wd_regularizer: float = 0.0,
        smoothing: float = 0.99,
        momentum: float = 0.0,
        nesterov: bool = False,
        centered: bool = False,
        eps: float = 1e-08,
        has_aux: bool = False,
        axis_name: Optional[str] = None,
        grad_mask: Optional[Callable[[Param], Param]] = None,
    ) -> Tuple[Any, RMSPropState]:
    """Updates state.

    Args:
        state: Current state.
        batch: It will be send to `loss_fn`.
        loss_fn: Loss function to be differentiated.
        learning_rate: Learning rate.
        l2_regularizer: L2 regularization coefficient.
        wd_regularizer: Decoupled weight decay regularization coefficient.
        smoothing: Smoothing coefficient.
        momentum: Momentum coefficient.
        nesterov: It specifies whether the gradient computation incorporates the
            Nesterov accelerated gradient method.
        centered: It specifies whether the gradient is normalized by an estimate
            of its variance.
        eps: Small value added to the denominator.
        has_aux: It indicates whether the `energy_fn` returns a pair, with the
            first element as the main output of the energy function for
            differentiation and the second element as optional auxiliary data.
        axis_name: `gradients` will be averaged across replicas if specified.
        grad_mask: It applies arbitrary transformation to `gradient`.

    Returns:
        Auxiliary data and updated state.
    """
    position = jax.tree_util.tree_map(
        lambda p, m: p - m * momentum,
        state.position, state.momentum) if nesterov else state.position

    aux, gradient = jax.value_and_grad(
        loss_fn, argnums=0, has_aux=has_aux)(position, batch)
    if l2_regularizer > 0:
        gradient = jax.tree_util.tree_map(
            lambda p, g: p * l2_regularizer + g,
            state.position, gradient)
    if axis_name is not None:
        gradient = jax.lax.pmean(gradient, axis_name)
    if grad_mask is not None:
        gradient = grad_mask(gradient)

    momentum_nu = jax.tree_util.tree_map(
        lambda nu, g: smoothing * nu + (1.0 - smoothing) * jnp.square(g),
        state.momentum_nu, gradient)

    if centered:
        momentum_mu = jax.tree_util.tree_map(
            lambda mu, g: smoothing * mu + (1.0 - smoothing) * g,
            state.momentum_mu, gradient)
        momentum_nu_hat = jax.tree_util.tree_map(
            lambda mu, nu: nu - jnp.square(mu),
            momentum_mu, momentum_nu)
    else:
        momentum_mu = state.momentum_mu
        momentum_nu_hat = momentum_nu

    if momentum > 0:
        momentum = jax.tree_util.tree_map(
            lambda m, g, nu: m * momentum + g / jnp.sqrt(nu + eps),
            state.momentum, gradient, momentum_nu_hat)
        position = jax.tree_util.tree_map(
            lambda p, m:
                (1.0 - wd_regularizer) * p
                - m * learning_rate,
            state.position, momentum)
    else:
        momentum = state.momentum
        position = jax.tree_util.tree_map(
            lambda p, g, nu:
                (1.0 - wd_regularizer) * p
                - g * learning_rate / jnp.sqrt(nu + eps),
            state.position, gradient, momentum_nu_hat)

    return aux, RMSPropState(
        step=state.step+1, position=position,
        momentum=momentum, momentum_mu=momentum_mu, momentum_nu=momentum_nu)
