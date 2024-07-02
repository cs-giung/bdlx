"""Utilities for handling pytree objects."""
import pickle
from pathlib import Path
from typing import Union

import jax
from bdlx.typing import PRNGKeyLike, Pytree, PytreeLike


def randn_like(rng_key: PRNGKeyLike, pytree: PytreeLike) -> Pytree:
    tree = jax.tree_util.tree_structure(pytree)
    keys = jax.tree_util.tree_unflatten(
        tree, jax.random.split(rng_key, tree.num_leaves))
    return jax.tree_util.tree_map(
        lambda p, k: jax.random.normal(k, p.shape, p.dtype), pytree, keys)


def save(
        file: Union[str, Path],
        pytree: PytreeLike,
        *,
        overwrite: bool = False,
    ) -> None:
    """Save a pytree to a binary file in `.pickle` format.

    Args:
        file: Filename to which the data is saved; a `.pickle` extension will
            be appended to the filename if it does not already ahve one.
        pytree: Pytree data to be saved.
        overwrite: It prohibits overwriting of files (default: False).
    """
    file = Path(file)
    if file.suffix != '.pickle':
        file = file.with_suffix('.pickle')

    if file.exists():
        if overwrite:
            file.unlink()
        else:
            raise RuntimeError(
                f'{file} already exists, while overwrite is {overwrite}.')

    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, 'wb') as fp:
        pickle.dump(pytree, fp)


def load(file: Union[str, Path]) -> Pytree:
    """Load a pytree from a binary file in `.pickle` format.

    Args:
        file: Filename to which the data is saved.
    """
    file = Path(file)
    if not file.is_file():
        raise ValueError(f'{file} is not a file.')
    if file.suffix != '.pickle':
        raise ValueError(f'{file} is not a .pickle file.')
    with open(file, 'rb') as fp:
        pytree = pickle.load(fp)
    return pytree
