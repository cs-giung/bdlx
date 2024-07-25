"""Mini-batch Stochastic Gradient Descent"""
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
from bdlx.typing import Pytree


Param = Pytree
Batch = Pytree


class SGDState(NamedTuple):
    """State including position."""
    step: int
    position: Pytree


def step( # pylint: disable=too-many-arguments,too-many-locals
        state: SGDState,
        batch: Batch,
        loss_fn: Callable[[Param, Batch], Any],
        learning_rate: float,
        l2_regularizer: float = 0.0,
        wd_regularizer: float = 0.0,
        has_aux: bool = False,
        axis_name: Optional[str] = None,
        grad_mask: Optional[Callable[[Param], Param]] = None,
    ) -> Tuple[Any, SGDState]:
    """Updates state.

    Args:
        state: Current state.
        batch: It will be send to `loss_fn`.
        loss_fn: Loss function to be differentiated.
        learning_rate: Learning rate.
        l2_regularizer: L2 regularization coefficient.
        wd_regularizer: Decoupled weight decay regularization coefficient.
        has_aux: It indicates whether the `energy_fn` returns a pair, with the
            first element as the main output of the energy function for
            differentiation and the second element as optional auxiliary data.
        axis_name: `gradients` will be averaged across replicas if specified.
        grad_mask: It applies arbitrary transformation to `gradient`.

    Returns:
        Auxiliary data and updated state.
    """
    aux, gradient = jax.value_and_grad(
        loss_fn, argnums=0, has_aux=has_aux)(state.position, batch)
    if l2_regularizer > 0:
        gradient = jax.tree_util.tree_map(
            lambda p, g: p * l2_regularizer + g,
            state.position, gradient)
    if axis_name is not None:
        gradient = jax.lax.pmean(gradient, axis_name)
    if grad_mask is not None:
        gradient = grad_mask(gradient)

    position = jax.tree_util.tree_map(
        lambda p, g: (1.0 - wd_regularizer) * p - learning_rate * g,
        state.position, gradient)

    return aux, SGDState(step=state.step+1, position=position)
