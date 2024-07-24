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


class SGRHMCDiagEFState(NamedTuple):
    """State including position and momentum."""
    step: int
    rng_key: PRNGKey
    position: Pytree
    momentum: Pytree
    momentum_nu: Pytree


def step(  # pylint: disable=too-many-arguments,too-many-locals
        state: SGRHMCDiagEFState,
        batch: Batch,
        num_train: int,
        perex_log_likelihood_fn: Callable[[Param, Batch], Any],
        perex_log_prior_fn: Callable[[Param, Batch], Any],
        step_size: float,
        smoothing: float = 0.999,
        gradient_noise: float = 0.0,
        damping_factor: float = 0.0,
        temperature: float = 1.0,
        has_aux: bool = False,
        axis_name: Optional[str] = None,
    ) -> Tuple[Any, SGRHMCDiagEFState]:
    """Updates state.

    Args:
        state: Current state.
        batch: It will be send to `energy_fn`.
        num_train:
        damping_factor:
        perex_log_likelihood_fn:
        perex_log_prior_fn:
        step_size: Step size, denoted by $\\epsilon$ in the paper. Note that
            `step_size**2 * train_size` corresponds to the learning rate in the
            conventional MomentumSGD implementation.
        gradient_noise: Gradient noise coefficient for non-tempered posterior.
        temperature: Temperature of joint distribution for posterior tempering.
        has_aux: It indicates whether the `energy_fn` returns a pair, with the
            first element as the main output of the energy function for
            differentiation and the second element as optional auxiliary data.
        axis_name: `gradients` will be averaged across replicas if specified.

    Returns:
        Auxiliary data and updated state.
    """
    aux_ll, gradient_ll = jax.vmap(
        jax.value_and_grad(
            perex_log_likelihood_fn, argnums=0, has_aux=has_aux
        ), in_axes=(None, 0))(state.position, batch)
    aux_lp, gradient_lp = jax.vmap(
        jax.value_and_grad(
            perex_log_prior_fn, argnums=0, has_aux=has_aux
        ), in_axes=(None, 0))(state.position, batch)
    if axis_name is not None:
        gradient_ll = jax.tree_util.tree_map(
            lambda e: e.reshape((-1,) + e.shape[2:]),
            jax.lax.all_gather(gradient_ll, axis_name))
        gradient_lp = jax.tree_util.tree_map(
            lambda e: e.reshape((-1,) + e.shape[2:]),
            jax.lax.all_gather(gradient_lp, axis_name))

    gradient = jax.tree_util.tree_map(
        lambda g_ll, g_lp:
            jnp.mean(g_ll + g_lp, axis=0) * num_train,
        gradient_ll, gradient_lp)
    momentum_nu = jax.tree_util.tree_map(
        lambda nu, g_ll: smoothing * nu + (1.0 - smoothing) * (
            jnp.mean(jnp.square(g_ll), axis=0) * num_train + damping_factor),
        state.momentum_nu, gradient_ll)

    noise = randn_like(state.rng_key, state.position)
    momentum = jax.tree_util.tree_map(
        lambda m, g, n, nu: \
            m * (1. - step_size / nu) \
            + g * step_size / jnp.sqrt(nu) \
            + n * jnp.sqrt(
                2. * step_size / nu * temperature
                - gradient_noise * step_size**2 * temperature**2),
        state.momentum, gradient, noise, momentum_nu)
    position = jax.tree_util.tree_map(
        lambda p, m, nu: p + m * step_size / jnp.sqrt(nu),
        state.position, momentum, momentum_nu)

    return (aux_ll[0] + aux_lp[0], {**aux_ll[1], **aux_lp[1]}), \
        SGRHMCDiagEFState(
            step=state.step+1, rng_key=jax.random.split(state.rng_key)[0],
            position=position, momentum=momentum, momentum_nu=momentum_nu)
