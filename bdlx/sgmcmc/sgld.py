"""
Bayesian Learning via Stochastic Gradient Langevin Dynamics
https://dl.acm.org/doi/10.5555/3104482.3104568
"""
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from bdlx.tree_util import randn_like
from bdlx.typing import PRNGKey, Pytree


Param = Pytree
Batch = Pytree


class SGLDState(NamedTuple):
    """State including position."""
    step: int
    rng_key: PRNGKey
    position: Pytree


def step(  # pylint: disable=too-many-arguments,too-many-locals
        state: SGLDState,
        batch: Batch,
        energy_fn: Callable[[Param, Batch], Any],
        step_size: float,
        temperature: float = 1.0,
        has_aux: bool = False,
        axis_name: Optional[str] = None,
        grad_mask: Optional[Callable[[Param], Param]] = None,
        ) -> Tuple[Any, SGLDState]:
    """Updates state.

    Args:
        state: Current state.
        batch: It will be send to `energy_fn`.
        energy_fn: Energy function to be differentiated; it should take
            `state.position` and `batch` and return the posterior energy value
            as well as auxiliary information.
        step_size: Step size, denoted by $\\epsilon / 2$ in the original paper.
        temperature: Temperature of joint distribution for posterior tempering.
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
    position = jax.tree_util.tree_map(
        lambda p, g, n:
            p - g * step_size + n * jnp.sqrt(2. * step_size * temperature),
        state.position, gradient, noise)

    return aux, SGLDState(
        step=state.step+1, rng_key=jax.random.split(state.rng_key)[0],
        position=position)
