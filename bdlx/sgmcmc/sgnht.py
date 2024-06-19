"""
Bayesian Sampling Using Stochastic Gradient Thermostats
https://dl.acm.org/doi/abs/10.5555/2969033.2969184
"""
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.flatten_util
import jax.numpy as jnp
from bdlx.tree_util import randn_like
from bdlx.typing import PRNGKey, Pytree


Param = Pytree
Batch = Pytree


class SGNHTState(NamedTuple):
    """State including position, momentum, and learnable friction."""
    step: int
    rng_key: PRNGKey
    position: Pytree
    momentum: Pytree
    friction: float


def step( # pylint: disable=too-many-arguments,too-many-locals
        state: SGNHTState,
        batch: Batch,
        energy_fn: Callable[[Param, Batch], Any],
        step_size: float,
        friction: float = None,
        momentum_decay: float = None,
        momentum_stdev: float = 1.0,
        temperature: float = 1.0,
        has_aux: bool = False,
        axis_name: Optional[str] = None,
        grad_mask: Optional[Callable[[Param], Param]] = None,
    ) -> Tuple[Any, SGNHTState]:
    """Updates state.

    Args:
        state: Current state.
        batch: It will be send to `energy_fn`.
        energy_fn: Energy function to be differentiated.
        step_size: Step size.
        friction: Friction coefficient.
        momentum_decay: Momentum decay coefficient.
        momentum_stdev: Standard deviation of momenta target distribution.
        temperature: Temperature of joint distribution for posterior tempering.
        has_aux: It indicates whether the `energy_fn` returns a pair, with the
            first element as the main output of the energy function for
            differentiation and the second element as optional auxiliary data.
        axis_name: `gradients` will be averaged across replicas if specified.
        grad_mask: It applies arbitrary transformation to `gradient`.

    Returns:
        Auxiliary data and updated state.
    """
    if friction is None and momentum_decay is None:
        raise AssertionError(
            'Either friction or momentum_decay must be specified.')
    if momentum_decay is None:
        momentum_decay = step_size * friction

    aux, gradient = jax.value_and_grad(
        energy_fn, argnums=0, has_aux=has_aux)(state.position, batch)
    if axis_name is not None:
        gradient = jax.lax.pmean(gradient, axis_name)
    if grad_mask is not None:
        gradient = grad_mask(gradient)

    noise = randn_like(state.rng_key, state.position)
    momentum = jax.tree_util.tree_map(
        lambda m, g, n: \
            m * (1. - step_size * state.friction / momentum_stdev**2) \
            + g * step_size \
            + n * (2. * momentum_decay * momentum_stdev**2 * temperature)**2,
        state.momentum, gradient, noise)
    position = jax.tree_util.tree_map(
        lambda p, m: p - m * step_size / momentum_stdev**2,
        state.position, momentum)
    friction = state.friction + step_size * (
        momentum_stdev**2 * jnp.mean(
            jax.flatten_util.ravel_pytree(state.momentum)[0]**2) - temperature)

    return aux, SGNHTState(
        step=state.step+1, rng_key=jax.random.split(state.rng_key)[0],
        position=position, momentum=momentum, friction=friction)
