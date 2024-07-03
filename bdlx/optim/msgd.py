"""
SGD with Momentum as well as Nesterov Accelerated Gradient
- https://doi.org/10.1038/323533a0
- https://zbmath.org/0535.90071
"""
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
from bdlx.typing import Pytree


Param = Pytree
Batch = Pytree


class MSGDState(NamedTuple):
    """State including position and momentum."""
    step: int
    position: Pytree
    momentum: Pytree


def step( # pylint: disable=too-many-arguments,too-many-locals
        state: MSGDState,
        batch: Batch,
        loss_fn: Callable[[Param, Batch], Any],
        learning_rate: float,
        l2_regularizer: float = 0.0,
        momentum: float = 0.9,
        nesterov: bool = False,
        has_aux: bool = False,
        axis_name: Optional[str] = None,
        grad_mask: Optional[Callable[[Param], Param]] = None,
    ) -> Tuple[Any, MSGDState]:
    """Updates state.

    Args:
        state: Current state.
        batch: It will be send to `loss_fn`.
        loss_fn: Loss function to be differentiated.
        learning_rate: Learning rate.
        l2_regularizer: L2 regularization coefficient.
        momentum: Momentum coefficient.
        nesterov: It specifies whether the gradient computation incorporates the
            Nesterov accelerated gradient method.
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
    if axis_name is not None:
        gradient = jax.lax.pmean(gradient, axis_name)
    if grad_mask is not None:
        gradient = grad_mask(gradient)

    momentum = jax.tree_util.tree_map(
        lambda p, m, g: \
            m * momentum + (g + p * l2_regularizer) * learning_rate,
        state.position, state.momentum, gradient)
    position = jax.tree_util.tree_map(
        lambda p, m: p - m,
        state.position, momentum)

    return aux, MSGDState(
        step=state.step+1, position=position, momentum=momentum)
