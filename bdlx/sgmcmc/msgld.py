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


class MSGLDState(NamedTuple):
    """State including position and momentum."""
    step: int
    rng_key: PRNGKey
    position: Pytree
    momentum: Pytree


def step( # pylint: disable=too-many-arguments,too-many-locals
        state: MSGLDState,
        batch: Batch,
        energy_fn: Callable[[Param, Batch], Any],
        step_size: float,
        smoothing: float = 0.9,
        bias: float = 1.0,
        temperature: float = 1.0,
        has_aux: bool = False,
        axis_name: Optional[str] = None,
        grad_mask: Optional[Callable[[Param], Param]] = None,
    ) -> Tuple[Any, MSGLDState]:
    """Updates state.

    Args:
        state: Current state.
        batch: It will be send to `energy_fn`.
        energy_fn: Energy function to be differentiated; it should take
            `state.position` and `batch` and return the posterior energy value
            as well as auxiliary information.
        step_size: Step size, denoted by $\\epsilon$ in the original paper.
        smoothing: Smoothing factor for the first moment of gradients.
        bias: Bias factor for the adaptive bias term.
        temperature: Temperature of joint distribution for posterior tempering.
            Setting `temperature` to zero is equivalent to MomentumSGD.
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

    noise = randn_like(state.rng_key, state.position)
    momentum = jax.tree_util.tree_map(
        lambda m, g: m * smoothing + g * (1. - smoothing),
        state.momentum, gradient)
    position = jax.tree_util.tree_map(
        lambda p, m, g, n: \
            p - step_size * (g + bias * m) \
            + n * jnp.sqrt(2. * step_size * temperature),
        state.position, momentum, gradient, noise)

    return aux, MSGLDState(
        step=state.step+1, rng_key=jax.random.split(state.rng_key)[0],
        position=position, momentum=momentum)
