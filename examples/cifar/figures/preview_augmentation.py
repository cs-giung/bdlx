"""Create preview images."""
import os
import sys
os.environ['JAX_PLATFORMS'] = 'cpu' # pylint: disable=wrong-import-position
sys.path.append('./') # pylint: disable=wrong-import-position

import jax
import matplotlib.pyplot as plt
import numpy as np

from examples.cifar import image_processing


trn_images = np.load('./examples/cifar/data/cifar10/train_images.npy')[:64]

_image = trn_images.reshape(8, 8, 32, 32, 3)
fig, ax = plt.subplots(nrows=8, ncols=8, figsize=(8, 8))
for i in range(8):
    for j in range(8):
        ax[i][j].imshow(_image.astype(np.uint8)[i][j])
        ax[i][j].set_xticks([])
        ax[i][j].set_yticks([])
plt.tight_layout()
plt.savefig('examples/cifar/figures/preview_none.png')

trn_augmentation = jax.jit(jax.vmap(image_processing.TransformChain([
    image_processing.RandomCropTransform(size=32, padding=4),
    image_processing.RandomHFlipTransform(prob=0.5)])))
_image = 255 * trn_augmentation(
    jax.random.split(jax.random.PRNGKey(0), 64),
    trn_images / 255.).reshape(8, 8, 32, 32, 3)
fig, ax = plt.subplots(nrows=8, ncols=8, figsize=(8, 8))
for i in range(8):
    for j in range(8):
        ax[i][j].imshow(_image.astype(np.uint8)[i][j])
        ax[i][j].set_xticks([])
        ax[i][j].set_yticks([])
plt.tight_layout()
plt.savefig('examples/cifar/figures/preview_simple.png')

trn_augmentation = jax.jit(jax.vmap(image_processing.TransformChain([
    image_processing.TransformChain([
        image_processing.RandomBrightnessTransform(0.8, 1.2),
        image_processing.RandomContrastTransform(0.8, 1.2),
        image_processing.RandomSaturationTransform(0.8, 1.2),
        image_processing.RandomHueTransform(0.1),
    ], prob=0.8),
    image_processing.RandomGrayscaleTransform(prob=0.2),
    image_processing.RandomCropTransform(size=32, padding=4),
    image_processing.RandomHFlipTransform(prob=0.5)])))
_image = 255 * trn_augmentation(
    jax.random.split(jax.random.PRNGKey(0), 64),
    trn_images / 255.).reshape(8, 8, 32, 32, 3)
fig, ax = plt.subplots(nrows=8, ncols=8, figsize=(8, 8))
for i in range(8):
    for j in range(8):
        ax[i][j].imshow(_image.astype(np.uint8)[i][j])
        ax[i][j].set_xticks([])
        ax[i][j].set_yticks([])
plt.tight_layout()
plt.savefig('examples/cifar/figures/preview_colour.png')
