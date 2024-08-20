"""Create preview images."""
import math
import os
import sys
os.environ['JAX_PLATFORMS'] = 'cpu' # pylint: disable=wrong-import-position
sys.path.append('./') # pylint: disable=wrong-import-position

import jax
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
import tensorflow_datasets
tensorflow.random.set_seed(0) # pylint: disable=wrong-import-position

from examples.imagenet import image_processing
from examples.imagenet.input_pipeline import create_trn_iter


shard_shape = (jax.local_device_count(), -1)

tfds_builder = tensorflow_datasets.builder(
    'imagenet2012', data_dir='~/tensorflow_datasets/')
image_decoder = tfds_builder.info.features['image'].decode_example

trn_dataset_size = tfds_builder.info.splits['train[:99%]'].num_examples
trn_steps_per_epoch = math.ceil(trn_dataset_size / 256)
trn_dataset = tfds_builder.as_dataset(
    split='train[:99%]', shuffle_files=True,
    decoders={'image': tensorflow_datasets.decode.SkipDecoding()})
trn_iter = create_trn_iter(
    trn_dataset, trn_dataset_size,
    image_decoder, 256, shard_shape)

trn_images = np.array(next(trn_iter)['images']).reshape(-1, 224, 224, 3)[:64]

_image = trn_images.reshape(8, 8, 224, 224, 3)
fig, ax = plt.subplots(nrows=8, ncols=8, figsize=(8, 8))
for i in range(8):
    for j in range(8):
        ax[i][j].imshow(_image.astype(np.uint8)[i][j])
        ax[i][j].set_xticks([])
        ax[i][j].set_yticks([])
plt.tight_layout()
plt.savefig('examples/imagenet2012/figures/preview_simple.png')

trn_augmentation = jax.jit(jax.vmap(image_processing.TransformChain([
    image_processing.TransformChain([
        image_processing.RandomBrightnessTransform(0.6, 1.4),
        image_processing.RandomContrastTransform(0.6, 1.4),
        image_processing.RandomSaturationTransform(0.6, 1.4),
        image_processing.RandomHueTransform(0.1),
    ], prob=0.8),
    image_processing.RandomGrayscaleTransform(prob=0.2),
    image_processing.TransformChain([
        image_processing.RandomGaussianBlurTransform(
            kernel_size=32, sigma=(0.1, 2.0)),], prob=0.5)])))
_image = 255 * trn_augmentation(
    jax.random.split(jax.random.PRNGKey(0), 64),
    trn_images / 255.).reshape(8, 8, 224, 224, 3)
fig, ax = plt.subplots(nrows=8, ncols=8, figsize=(8, 8))
for i in range(8):
    for j in range(8):
        ax[i][j].imshow(_image.astype(np.uint8)[i][j])
        ax[i][j].set_xticks([])
        ax[i][j].set_yticks([])
plt.tight_layout()
plt.savefig('examples/imagenet2012/figures/preview_colour.png')


# trn_augmentation = jax.jit(jax.vmap(image_processing.TransformChain([
#     image_processing.RandomCropTransform(size=32, padding=4),
#     image_processing.RandomHFlipTransform(prob=0.5)])))
# _image = 255 * trn_augmentation(
#     jax.random.split(jax.random.PRNGKey(0), 64),
#     trn_images / 255.).reshape(8, 8, 32, 32, 3)
# fig, ax = plt.subplots(nrows=8, ncols=8, figsize=(8, 8))
# for i in range(8):
#     for j in range(8):
#         ax[i][j].imshow(_image.astype(np.uint8)[i][j])
#         ax[i][j].set_xticks([])
#         ax[i][j].set_yticks([])
# plt.tight_layout()
# plt.savefig('examples/cifar/figures/preview_simple.png')

# trn_augmentation = jax.jit(jax.vmap(image_processing.TransformChain([
#     image_processing.TransformChain([
#         image_processing.RandomBrightnessTransform(0.8, 1.2),
#         image_processing.RandomContrastTransform(0.8, 1.2),
#         image_processing.RandomSaturationTransform(0.8, 1.2),
#         image_processing.RandomHueTransform(0.1),
#     ], prob=0.8),
#     image_processing.RandomGrayscaleTransform(prob=0.2),
#     image_processing.RandomCropTransform(size=32, padding=4),
#     image_processing.RandomHFlipTransform(prob=0.5)])))
# _image = 255 * trn_augmentation(
#     jax.random.split(jax.random.PRNGKey(0), 64),
#     trn_images / 255.).reshape(8, 8, 32, 32, 3)
# fig, ax = plt.subplots(nrows=8, ncols=8, figsize=(8, 8))
# for i in range(8):
#     for j in range(8):
#         ax[i][j].imshow(_image.astype(np.uint8)[i][j])
#         ax[i][j].set_xticks([])
#         ax[i][j].set_yticks([])
# plt.tight_layout()
# plt.savefig('examples/cifar/figures/preview_colour.png')
