"""
SAM as an Optimal Relaxation of Bayes
https://arxiv.org/abs/2210.01620
"""
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from bdlx.tree_util import randn_like
from bdlx.typing import PRNGKey, Pytree


Param = Pytree
Batch = Pytree


class bSAMState(NamedTuple):
    """State including position and momentum."""
    step: int
    rng_key: PRNGKey
    position: Pytree
    momentum_mu: Pytree
    momentum_nu: Pytree


def step( # pylint: disable=too-many-arguments,too-many-locals
        state: bSAMState,
        batch: Batch,
        loss_fn: Callable[[Param, Batch], Any],
        effective_sample_size: float,
        learning_rate: float,
        radius: float,
        l2_regularizer: float,
        wd_regularizer: float = 0.0,
        momentums: Tuple[float, float] = (0.9, 0.99999),
        eps: float = 0.1,
        clip_radius: float = None,
        has_aux: bool = False,
        axis_name: Optional[str] = None,
        grad_mask: Optional[Callable[[Param], Param]] = None,
    ) -> Tuple[Any, bSAMState]:
    """Updates state.

    Args:
        state: Current state.
        batch: It will be send to `loss_fn`.
        loss_fn: Loss function to be differentiated.
        effective_sample_size: Effective sample size.
        learning_rate: Learning rate.
        radius: a float radius value for sharpness-aware minimization.
        l2_regularizer: L2 regularization coefficient.
        wd_regularizer: Decoupled weight decay regularization coefficient.
        momentums: Momentum coefficients.
        eps: a float constant value for improving variance.
        clip_radius: Clip the preconditioned gradient if specified.
        has_aux: It indicates whether the `energy_fn` returns a pair, with the
            first element as the main output of the energy function for
            differentiation and the second element as optional auxiliary data.
        axis_name: `gradients` will be averaged across replicas if specified.
        grad_mask: It applies arbitrary transformation to `gradient`.

    Returns:
        Auxiliary data and updated state.
    """
    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=has_aux)

    noise = randn_like(state.rng_key, state.position)
    noise = jax.tree_util.tree_map(
        lambda n, nu: n * jnp.sqrt(
            1.0 / (effective_sample_size * (nu + l2_regularizer))),
        noise, state.momentum_nu)
    if grad_mask is not None:
        noise = grad_mask(noise)
    noisy_position = jax.tree_util.tree_map(
        lambda p, n: p + n,
        state.position, noise)

    aux, gradient = grad_fn(noisy_position, batch)
    if axis_name is not None:
        gradient = jax.lax.pmean(gradient, axis_name)
    if grad_mask is not None:
        gradient = grad_mask(gradient)

    adv_position = jax.tree_util.tree_map(
        lambda p, g, nu: p + g * radius / nu,
        state.position, gradient, state.momentum_nu)
    adv_gradient = grad_fn(adv_position, batch)[1]
    if axis_name is not None:
        adv_gradient = jax.lax.pmean(adv_gradient, axis_name)
    if grad_mask is not None:
        adv_gradient = grad_mask(adv_gradient)

    momentum_mu = jax.tree_util.tree_map(
        lambda mu, g, p:
            momentums[0] * mu + (1.0 - momentums[0]) * (g + p * l2_regularizer),
        state.momentum_mu, adv_gradient, state.position)
    momentum_nu = jax.tree_util.tree_map(
        lambda nu, g:
            momentums[1] * nu + (1.0 - momentums[1]) * (
                jnp.sqrt(nu) * jnp.abs(g) + l2_regularizer + eps),
        state.momentum_nu, gradient)

    # pylint: disable=invalid-unary-operand-type
    updates = jax.tree_util.tree_map(
        lambda mu, nu: mu / nu,
        momentum_mu, momentum_nu)
    if clip_radius is not None:
        updates = jax.tree_util.tree_map(
            lambda u: jnp.clip(u, -clip_radius, clip_radius),
            updates)
    if grad_mask is not None:
        updates = grad_mask(updates)

    position = jax.tree_util.tree_map(
        lambda p, u: (1.0 - wd_regularizer) * p - learning_rate * u,
        state.position, updates)

    return aux, bSAMState(
        step=state.step+1, rng_key=jax.random.split(state.rng_key)[0],
        position=position, momentum_mu=momentum_mu, momentum_nu=momentum_nu)
