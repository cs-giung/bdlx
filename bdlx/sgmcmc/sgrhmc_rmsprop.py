"""
A Complete Recipe for Stochastic Gradient MCMC
https://dl.acm.org/doi/abs/10.5555/2969442.2969566
"""
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from bdlx.tree_util import randn_like
from bdlx.typing import PRNGKey, Pytree


Param = Pytree
Batch = Pytree


class SGRHMCRMSPropState(NamedTuple):
    """State including position and momentums."""
    step: int
    rng_key: PRNGKey
    position: Pytree
    momentum: Pytree
    momentum_nu: Pytree


def step(  # pylint: disable=too-many-arguments,too-many-locals
        state: SGRHMCRMSPropState,
        batch: Batch,
        energy_fn: Callable[[Param, Batch], Any],
        step_size: float,
        smoothing: float = 0.999,
        gradient_noise: float = 0.0,
        temperature: float = 1.0,
        eps: float = 1e-08,
        has_aux: bool = False,
        axis_name: Optional[str] = None,
        grad_mask: Optional[Callable[[Param], Param]] = None,
        ) -> Tuple[Any, SGRHMCRMSPropState]:
    """Updates state.

    Args:
        state: Current state.
        batch: It will be send to `energy_fn`.
        energy_fn: Energy function to be differentiated; it should take
            `state.position` and `batch` and return the posterior energy value
            as well as auxiliary information.
        step_size: Step size.
        smoothing: Smoothing coefficient.
        gradient_noise: Gradient noise coefficient for non-tempered posterior.
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

    momentum_nu = jax.tree_util.tree_map(
        lambda nu, g: smoothing * nu + (1.0 - smoothing) * jnp.square(g),
        state.momentum_nu, gradient)

    noise = randn_like(state.rng_key, state.position)
    momentum = jax.tree_util.tree_map(
        lambda m, nu, g, n:
            m * (1. - step_size / (nu + eps))
            + g * step_size / jnp.sqrt(nu + eps)
            + n * jnp.sqrt(
                2. * step_size / (nu + eps) * temperature
                - gradient_noise * step_size**2 * temperature**2),
        state.momentum, state.momentum_nu, gradient, noise)
    position = jax.tree_util.tree_map(
        lambda p, m, nu: p - m * step_size / jnp.sqrt(nu + eps),
        state.position, momentum, momentum_nu)

    return aux, SGRHMCRMSPropState(
        step=state.step+1, rng_key=jax.random.split(state.rng_key)[0],
        position=position, momentum=momentum, momentum_nu=momentum_nu)
