"""
Stochastic Gradient Langevin Dynamics Algorithms with Adaptive Drifts
https://arxiv.org/abs/2009.09535
"""
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from bdlx.tree_util import randn_like
from bdlx.typing import PRNGKey, Pytree


Param = Pytree
Batch = Pytree


class ASGLDState(NamedTuple):
    """State including position and momentums."""
    step: int
    rng_key: PRNGKey
    position: Pytree
    momentum_mu: Pytree
    momentum_nu: Pytree


def step(  # pylint: disable=too-many-arguments,too-many-locals
        state: ASGLDState,
        batch: Batch,
        energy_fn: Callable[[Param, Batch], Any],
        step_size: float,
        smoothing: Tuple[float, float] = (0.9, 0.999),
        bias: float = 1.0,
        temperature: float = 1.0,
        eps: float = 1e-08,
        has_aux: bool = False,
        axis_name: Optional[str] = None,
        grad_mask: Optional[Callable[[Param], Param]] = None,
        ) -> Tuple[Any, ASGLDState]:
    """Updates state.

    Args:
        state: Current state.
        batch: It will be send to `energy_fn`.
        energy_fn: Energy function to be differentiated; it should take
            `state.position` and `batch` and return the posterior energy value
            as well as auxiliary information.
        step_size: Step size, denoted by $\\epsilon$ in the original paper.
        smoothing: Smoothing coefficients.
        bias: Bias factor for the adaptive bias term.
        temperature: Temperature of joint distribution for posterior tempering.
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
        energy_fn, argnums=0, has_aux=has_aux)(state.position, batch)
    if axis_name is not None:
        gradient = jax.lax.pmean(gradient, axis_name)
    if grad_mask is not None:
        gradient = grad_mask(gradient)

    momentum_mu = jax.tree_util.tree_map(
        lambda mu, g: smoothing[0] * mu + (1.0 - smoothing[0]) * g,
        state.momentum_mu, gradient)
    momentum_nu = jax.tree_util.tree_map(
        lambda nu, g: smoothing[1] * nu + (1.0 - smoothing[1]) * jnp.square(g),
        state.momentum_nu, gradient)

    noise = randn_like(state.rng_key, state.position)
    position = jax.tree_util.tree_map(
        lambda p, mu, nu, g, n:
            p - step_size * (g + bias * mu / jnp.sqrt(nu + eps))
            + n * jnp.sqrt(2. * step_size * temperature),
        state.position, momentum_mu, momentum_nu, gradient, noise)

    return aux, ASGLDState(
        step=state.step+1, rng_key=jax.random.split(state.rng_key)[0],
        position=position, momentum_mu=momentum_mu, momentum_nu=momentum_nu)
