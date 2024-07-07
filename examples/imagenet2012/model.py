"""
What Are Bayesian Neural Network Posteriors Really Like?
https://arxiv.org/abs/2104.14421
"""
from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.dtypes import canonicalize_dtype


Dtype = Any


class FilterResponseNorm(nn.Module):
    """
    Filter Response Normalization Layer
    https://arxiv.org/abs/1911.09737
    """
    epsilon: float = 1e-6
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable = jax.nn.initializers.zeros
    scale_init: Callable = jax.nn.initializers.ones
    threshold_init: Callable = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs):
        y = inputs
        nu2 = jnp.mean(jnp.square(inputs), axis=(1, 2), keepdims=True)
        mul = jax.lax.rsqrt(nu2 + self.epsilon)
        if self.use_scale:
            scale = self.param(
                'scale', self.scale_init, (inputs.shape[-1],),
                self.param_dtype).reshape((1, 1, 1, -1))
            mul *= scale
        y *= mul
        if self.use_bias:
            bias = self.param(
                'bias', self.bias_init, (inputs.shape[-1],),
                self.param_dtype).reshape((1, 1, 1, -1))
            y += bias
        tau = self.param(
            'threshold', self.threshold_init, (inputs.shape[-1],),
            self.param_dtype).reshape((1, 1, 1, -1))
        z = jnp.maximum(y, tau)
        dtype = canonicalize_dtype(scale, bias, tau, dtype=self.dtype)
        return jnp.asarray(z, dtype)


class ResNet20x8(nn.Module):
    """
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """
    conv: nn.Module = partial(
        nn.Conv, use_bias=True,
        kernel_init=jax.nn.initializers.he_normal(),
        bias_init=jax.nn.initializers.zeros)
    norm: nn.Module = FilterResponseNorm
    relu: Callable = jax.nn.silu

    @nn.compact
    def __call__(self, x):
        # pylint: disable=too-many-function-args

        y = self.conv(features=128, kernel_size=(7, 7), strides=(2, 2))(x)
        y = self.norm()(y)
        y = self.relu(y)
        y = nn.max_pool(y, window_shape=(3, 3), stride=(2, 2), padding='SAME')

        for layer_idx, num_block in enumerate([3, 3, 3]):
            _strides = (1,) if layer_idx == 0 else (2,)
            _strides = _strides + (1,) * (num_block - 1)

            for s in _strides:
                _channel = 128 * (2 ** layer_idx)
                residual = y

                y = self.conv(
                    features=_channel, kernel_size=(3, 3), strides=(s, s))(y)
                y = self.norm()(y)
                y = self.relu(y)

                y = self.conv(
                    features=_channel, kernel_size=(3, 3), strides=(1, 1))(y)
                y = self.norm()(y)

                if residual.shape != y.shape:
                    residual = self.conv(
                        features=y.shape[-1],
                        kernel_size=(1, 1), strides=(s, s))(residual)
                    residual = self.norm()(residual)

                y = self.relu(y + residual)

        y = jnp.mean(y, axis=(1, 2))

        return y
