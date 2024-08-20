"""It provides image augmentations."""
from typing import List, Tuple
from abc import ABCMeta, abstractmethod

import jax
import jax.numpy as jnp
from bdlx.typing import ArrayLike, Array


def split_channels(
        image: ArrayLike,
        channel_axis: int,
    ) -> Tuple[Array, Array, Array]:
    """Splits an image into its channels."""
    split_axes = jnp.split(image, 3, axis=channel_axis)
    return tuple(map(lambda e: jnp.squeeze(e, axis=channel_axis), split_axes))


def rgb_planes_to_hsv_planes(
        r: ArrayLike, g: ArrayLike, b: ArrayLike
    ) -> Tuple[Array, Array, Array]:
    """Converts RGB color planes to HSV planes."""
    v = jnp.maximum(jnp.maximum(r, g), b)
    c = v - jnp.minimum(jnp.minimum(r, g), b)
    safe_v = jnp.where(v > 0., v, 1.)
    safe_c = jnp.where(c > 0., c, 1.)

    s = jnp.where(v > 0., c / safe_v, 0.)

    rc = (v - r) / safe_c
    gc = (v - g) / safe_c
    bc = (v - b) / safe_c

    h = jnp.where(r == v, bc - gc, 0.)
    h = jnp.where(g == v, 2. + rc - bc, h)
    h = jnp.where(b == v, 4. + gc - rc, h)

    h = (h / 6.) % 1.
    h = jnp.where(c == 0., 0., h)

    return h, s, v


def hsv_planes_to_rgb_planes(
        h: ArrayLike, s: ArrayLike, v: ArrayLike
    ) -> Tuple[Array, Array, Array]:
    """Converts HSV color planes to RGB planes."""
    h = h % 1.0
    dh = h * 6.0

    dr = jnp.clip(jnp.abs(dh - 3.) - 1., 0., 1.)
    dg = jnp.clip(2. - jnp.abs(dh - 2.), 0., 1.)
    db = jnp.clip(2. - jnp.abs(dh - 4.), 0., 1.)

    ms = 1. - s
    r = v * (ms + s * dr)
    g = v * (ms + s * dg)
    b = v * (ms + s * db)

    return r, g, b


def rgb_to_hsv(
        image: ArrayLike,
        *,
        channel_axis: int = -1,
    ) -> Array:
    """Converts an image from RGB to HSV."""
    r, g, b = split_channels(image, channel_axis)
    return jnp.stack(rgb_planes_to_hsv_planes(r, g, b), axis=channel_axis)


def hsv_to_rgb(
        image: ArrayLike,
        *,
        channel_axis: int = -1,
    ) -> Array:
    """Converts an image from HSV to RGB."""
    h, s, v = split_channels(image, channel_axis)
    return jnp.stack(hsv_planes_to_rgb_planes(h, s, v), axis=channel_axis)


def rgb_to_grayscale(
        image: ArrayLike,
        *,
        keepdims: bool = True,
        luma_standard='rec601',
        channel_axis: int = -1,
    ) -> Array:
    """Converts an image to a grayscale image."""
    if luma_standard == 'rec601':
        weight = jnp.array([0.2989, 0.5870, 0.1140], dtype=image.dtype)
    elif luma_standard == 'rec709':
        weight = jnp.array([0.2126, 0.7152, 0.0722], dtype=image.dtype)
    elif luma_standard == 'bt2001':
        weight = jnp.array([0.2627, 0.6780, 0.0593], dtype=image.dtype)
    else:
        raise NotImplementedError(f'Unknown luma_standard={luma_standard}')
    grayscale = jnp.tensordot(image, weight, axes=(channel_axis, -1))
    grayscale = jnp.expand_dims(grayscale, axis=channel_axis)
    if keepdims:
        if channel_axis < 0:
            channel_axis += image.ndim
        return jnp.tile(grayscale, [
            (1 if axis != channel_axis else 3) for axis in range(image.ndim)])
    return grayscale


class Transform(metaclass=ABCMeta):
    # pylint: disable=too-few-public-methods
    """Base class for transformations."""

    @abstractmethod
    def __call__(self, rng, image):
        """Apply the transform on an image."""


class TransformChain(Transform):
    # pylint: disable=too-few-public-methods
    """Chain multiple transformations."""

    def __init__(self, transforms: List[Transform], prob: float = 1.0):
        """Apply transforms with the given probability."""
        self.transforms = transforms
        self.prob = prob

    def __call__(self, rng, image):
        jmage = image
        _rngs = jax.random.split(rng, len(self.transforms))
        for _transform, _rng in zip(self.transforms, _rngs):
            jmage = _transform(_rng, jmage)
        return jnp.where(jax.random.bernoulli(rng, self.prob), jmage, image)


class RandomGrayscaleTransform(Transform):
    # pylint: disable=too-few-public-methods
    """RandomGrayscaleTransform"""

    def __init__(self, prob=0.5):
        """Grayscales an image with the given probability."""
        self.prob = prob

    def __call__(self, rng, image):
        jmage = rgb_to_grayscale(image, keepdims=True)
        jmage = jnp.clip(jmage, 0., 1.).astype(image.dtype)
        return jnp.where(
            jax.random.bernoulli(rng, self.prob), jmage, image)


class RandomBrightnessTransform(Transform):
    # pylint: disable=too-few-public-methods
    """RandomBrightnessTransform"""

    def __init__(self, lower=0.5, upper=1.5):
        """Changes a brightness of an image."""
        self.lower = lower
        self.upper = upper

    def __call__(self, rng, image):
        alpha = jax.random.uniform(
            rng, shape=(1,), minval=self.lower, maxval=self.upper)
        jmage = (image * alpha).astype(image.dtype)
        return jnp.clip(jmage, 0., 1.)


class RandomContrastTransform(Transform):
    # pylint: disable=too-few-public-methods
    """RandomContrastTransform"""

    def __init__(self, lower=0.5, upper=1.5):
        """Changes a contrast of an image."""
        self.lower = lower
        self.upper = upper

    def __call__(self, rng, image):
        alpha = jax.random.uniform(
            rng, shape=(1,), minval=self.lower, maxval=self.upper)
        _mean = jnp.mean(image, axis=(0, 1), keepdims=True)
        jmage = (_mean + (image - _mean) * alpha).astype(image.dtype)
        return jnp.clip(jmage, 0., 1.)


class RandomSaturationTransform(Transform):
    # pylint: disable=too-few-public-methods
    """RandomSaturationTransform"""

    def __init__(self, lower=0.5, upper=1.5):
        """Changes a saturation of an image."""
        self.lower = lower
        self.upper = upper

    def __call__(self, rng, image):
        alpha = jax.random.uniform(
            rng, shape=(1,), minval=self.lower, maxval=self.upper)
        r, g, b = split_channels(image, channel_axis=2)
        h, s, v = rgb_planes_to_hsv_planes(r, g, b)
        jmage = hsv_planes_to_rgb_planes(h, jnp.clip(s * alpha, 0., 1.), v)
        jmage = jnp.stack(jmage, axis=2).astype(image.dtype)
        return jnp.clip(jmage, 0., 1.)


class RandomHueTransform(Transform):
    # pylint: disable=too-few-public-methods
    """RandomHueTransform"""

    def __init__(self, delta=0.5):
        """Changes a hue of an image."""
        self.delta = delta

    def __call__(self, rng, image):
        alpha = jax.random.uniform(
            rng, shape=(1,), minval=-self.delta, maxval=self.delta)
        r, g, b = split_channels(image, channel_axis=2)
        h, s, v = rgb_planes_to_hsv_planes(r, g, b)
        jmage = hsv_planes_to_rgb_planes((h + alpha) % 1.0, s, v)
        jmage = jnp.stack(jmage, axis=2).astype(image.dtype)
        return jnp.clip(jmage, 0., 1.)


class RandomGaussianBlurTransform(Transform):
    # pylint: disable=too-few-public-methods
    """RandomGaussianBlurTransform"""

    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, rng, image):
        alpha = jax.random.uniform(
            rng, shape=(), minval=self.sigma[0], maxval=self.sigma[1])
        radius = int(self.kernel_size / 2)
        kernel_size = 2 * radius + 1

        blur_filter = jnp.exp(jnp.negative(jnp.arange(
            -radius, radius + 1).astype(jnp.float32)**2 / (2. * alpha**2)))
        blur_filter = blur_filter / jnp.sum(blur_filter)
        blur_v = jnp.tile(
            jnp.reshape(blur_filter, [kernel_size, 1, 1, 1]), [1, 1, 1, 3])
        blur_h = jnp.tile(
            jnp.reshape(blur_filter, [1, kernel_size, 1, 1]), [1, 1, 1, 3])

        jmage = image[jnp.newaxis, ...]
        jmage = jax.lax.conv_general_dilated(
            jmage, blur_h, (1, 1), 'SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=jmage.shape[3])
        jmage = jax.lax.conv_general_dilated(
            jmage, blur_v, (1, 1), 'SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=jmage.shape[3])
        jmage = jnp.squeeze(jmage, axis=0).astype(image.dtype)
        return jnp.clip(jmage, 0., 1.)
