"""
Variational Learning is Effective for Large Deep Networks
https://arxiv.org/abs/2402.17641
"""
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from bdlx.tree_util import randn_like
from bdlx.typing import PRNGKey, Pytree


Param = Pytree
Batch = Pytree


class IVONState(NamedTuple):
    """State including position and momentum."""
    step: int
    rng_key: PRNGKey
    position: Pytree
    momentum_mu: Pytree
    momentum_nu: Pytree


def step( # pylint: disable=too-many-arguments,too-many-locals
        state: IVONState,
        batch: Batch,
        loss_fn: Callable[[Param, Batch], Any],
        effective_sample_size: float,
        learning_rate: float,
        l2_regularizer: float,
        wd_regularizer: float = 0.0,
        momentums: Tuple[float, float] = (0.9, 0.99999),
        clip_radius: float = None,
        has_aux: bool = False,
        axis_name: Optional[str] = None,
        grad_mask: Optional[Callable[[Param], Param]] = None,
    ) -> Tuple[Any, IVONState]:
    """Updates state.

    Args:
        state: Current state.
        batch: It will be send to `loss_fn`.
        loss_fn: Loss function to be differentiated.
        effective_sample_size: Effective sample size.
        learning_rate: Learning rate.
        l2_regularizer: L2 regularization coefficient.
        wd_regularizer: Decoupled weight decay regularization coefficient.
        momentums: Momentum coefficients.
        clip_radius: Clip the preconditioned gradient if specified.
        has_aux: It indicates whether the `energy_fn` returns a pair, with the
            first element as the main output of the energy function for
            differentiation and the second element as optional auxiliary data.
        axis_name: `gradients` will be averaged across replicas if specified.
        grad_mask: It applies arbitrary transformation to `gradient`.

    Returns:
        Auxiliary data and updated state.
    """
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

    aux, gradient = jax.value_and_grad(
        loss_fn, argnums=0, has_aux=has_aux)(noisy_position, batch)
    if axis_name is not None:
        gradient = jax.lax.pmean(gradient, axis_name)
    if grad_mask is not None:
        gradient = grad_mask(gradient)

    momentum_mu = jax.tree_util.tree_map(
        lambda mu, g: momentums[0] * mu + (1.0 - momentums[0]) * g,
        state.momentum_mu, gradient)
    momentum_mu_hat = jax.tree_util.tree_map(
        lambda mu: mu / (1.0 - momentums[0] ** (state.step + 1)),
        momentum_mu)

    hess = jax.tree_util.tree_map(
        lambda g, n, nu: g * n * effective_sample_size * (nu + l2_regularizer),
        gradient, noise, state.momentum_nu)
    momentum_nu = jax.tree_util.tree_map(
        lambda nu, h: nu * momentums[1] + h * (1.0 - momentums[1]),
        state.momentum_nu, hess)

    # pylint: disable=invalid-unary-operand-type
    updates = jax.tree_util.tree_map(
        lambda p, mu, nu: (mu + l2_regularizer * p) / (nu + l2_regularizer),
        state.position, momentum_mu_hat, momentum_nu)
    if clip_radius is not None:
        updates = jax.tree_util.tree_map(
            lambda u: jnp.clip(u, -clip_radius, clip_radius),
            updates)
    if grad_mask is not None:
        updates = grad_mask(updates)

    position = jax.tree_util.tree_map(
        lambda p, u: (1.0 - wd_regularizer) * p - learning_rate * u,
        state.position, updates)

    return aux, IVONState(
        step=state.step+1, rng_key=jax.random.split(state.rng_key)[0],
        position=position, momentum_mu=momentum_mu, momentum_nu=momentum_nu)
