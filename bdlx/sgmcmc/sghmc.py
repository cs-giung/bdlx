"""
Stochastic Gradient Hamiltonian Monte Carlo
https://dl.acm.org/doi/abs/10.5555/3044805.3045080
"""
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from bdlx.tree_util import randn_like
from bdlx.typing import PRNGKey, Pytree


Param = Pytree
Batch = Pytree


class SGHMCState(NamedTuple):
    """State including position and momentum."""
    step: int
    rng_key: PRNGKey
    position: Pytree
    momentum: Pytree


def step( # pylint: disable=too-many-arguments,too-many-locals
        state: SGHMCState,
        batch: Batch,
        energy_fn: Callable[[Param, Batch], Any],
        step_size: float,
        friction: float = None,
        momentum_decay: float = None,
        momentum_stdev: float = 1.0,
        gradient_noise: float = 0.0,
        temperature: float = 1.0,
        has_aux: bool = False,
        axis_name: Optional[str] = None,
        grad_mask: Optional[Callable[[Param], Param]] = None,
    ) -> Tuple[Any, SGHMCState]:
    """Updates position and momentum.

    Args:
        state: `SGHMCState`.
        batch: It will be send to `energy_fn`.
        energy_fn: Energy function to be differentiated; it should take
            `state.position` and `batch` and return the posterior energy value
            as well as auxiliary information.
        step_size: Step size, denoted by $\\sqrt{\\eta}$ in the original paper.
            Note that `step_size**2 * train_size` corresponds to `lr` in
            MomentumSGD.
        friction: Friction coefficient, denoted by $\\alpha$ in the original
            paper. For the MomentumSGD parameterization, set `momentum_decay`
            coefficient instead.
        momentum_decay: Set `friction` to `momentum_decay / step_size` if
            specified. Note that `(1.0 - momentum_decay)` corresponds to
            `momentum` in MomentumSGD.
        momentum_stdev: Standard deviation of momenta target distribution.
        gradient_noise: Gradient noise coefficient, denoted by $\\beta$ in the
            original paper.
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
            m * (1. - momentum_stdev**2 * momentum_decay) \
            + g * step_size \
            + n * jnp.sqrt(
                2. * momentum_decay * temperature
                - gradient_noise * step_size**2 * temperature**2),
        state.momentum, gradient, noise)
    position = jax.tree_util.tree_map(
        lambda p, m: p - m * step_size / momentum_stdev**2,
        state.position, momentum)

    return aux, SGHMCState(
        step=state.step+1, rng_key=jax.random.split(state.rng_key)[0],
        position=position, momentum=momentum)
