"""Run optimization via stochastic gradient methods."""
import math
import os
import sys
from argparse import ArgumentParser
from collections import OrderedDict
from functools import partial
sys.path.append('./') # pylint: disable=wrong-import-position

import flax
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import tensorflow
import tensorflow_datasets
import wandb

from bdlx.optim import msgd
from bdlx.tree_util import load, save
from examples.imagenet import image_processing
from examples.imagenet.default import get_args, str2bool
from examples.imagenet.input_pipeline import \
    create_trn_iter, create_val_iter
from examples.imagenet.model import ResNet20x8


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument(
        '--ckpt', default=None, type=str,
        help='initialize position from *.ckpt if specified (default: None)')

    parser.add_argument(
        '--data_root', default='~/tensorflow_datasets/', type=str,
        help='root directory containing dataset files')
    parser.add_argument(
        '--data_name', default='imagenet2012', type=str,
        help='dataset name (default: imagenet2012)')
    parser.add_argument(
        '--data_augmentation', default='simple', type=str,
        help='apply data augmentation during training (default: simple)')

    parser.add_argument(
        '--num_samples', default=1, type=int,
        help='the number of samples (default: 1)')
    parser.add_argument(
        '--num_updates', default=500000, type=int,
        help='the number of updates for each sample (default: 500000)')
    parser.add_argument(
        '--num_batch', default=256, type=int,
        help='the number of instances in mini-batch (default: 256)')

    parser.add_argument(
        '--optim_lr', default=0.1, type=float,
        help='base learning rate (default: 0.1)')
    parser.add_argument(
        '--optim_lr_min', default=None, type=float,
        help='decay learning rate if specified (default: None)')

    parser.add_argument(
        '--optim_l2', default=0.0, type=float,
        help='regularization towards zero (default: 0.0)')
    parser.add_argument(
        '--optim_wd', default=0.0, type=float,
        help='regularization towards zero in decoupled manner (default: 0.0)')

    parser.add_argument(
        '--momentum', default=0.9, type=float,
        help='momentum coefficient (default: 0.9)')
    parser.add_argument(
        '--nesterov', default=False, type=str2bool,
        help='use Nesterov accelerated gradient (default: False)')

    parser.add_argument(
        '--use_wandb', default=False, type=str2bool,
        help='use wandb if specified (default: False)')

    args, print_fn, time_stamp = get_args(
        parser, exist_ok=True, dot_log_file=False,
        libraries=(flax, jax, jaxlib, tensorflow, tensorflow_datasets))

    if args.save == 'save/auto':
        args.save = os.path.join('save/auto', f'{time_stamp}_{args.seed}')
        os.makedirs(args.save, exist_ok=True)

    if args.optim_lr_min is None:
        args.optim_lr_min = args.optim_lr

    if args.use_wandb:
        wandb.init(project='bdlx', entity='cs-giung')
        wandb.config.update(args)

    # ----------------------------------------------------------------------- #
    # Data
    # ----------------------------------------------------------------------- #
    shard_shape = (jax.local_device_count(), -1)
    input_shape = (224, 224, 3)
    num_classes = 1000

    tfds_builder = tensorflow_datasets.builder(
        args.data_name, data_dir=args.data_root)
    image_decoder = tfds_builder.info.features['image'].decode_example
    trn_split = 'train[:99%]'
    dev_split = 'train[99%:]'
    val_split = 'validation'

    trn_dataset_size = tfds_builder.info.splits[trn_split].num_examples
    trn_steps_per_epoch = math.ceil(trn_dataset_size / args.num_batch)
    trn_dataset = tfds_builder.as_dataset(
        split=trn_split, shuffle_files=True,
        decoders={'image': tensorflow_datasets.decode.SkipDecoding()})
    trn_iter = create_trn_iter(
        trn_dataset, trn_dataset_size,
        image_decoder, args.num_batch, shard_shape)
    log_str = (
        f'It will go through {trn_steps_per_epoch} steps to handle '
        f'{trn_dataset_size} data for training.')
    print_fn(log_str)

    dev_dataset_size = tfds_builder.info.splits[dev_split].num_examples
    dev_steps_per_epoch = math.ceil(dev_dataset_size / args.num_batch)
    dev_dataset = tfds_builder.as_dataset(
        split=dev_split, shuffle_files=False,
        decoders={'image': tensorflow_datasets.decode.SkipDecoding()})
    dev_iter = create_val_iter(
        dev_dataset, dev_dataset_size,
        image_decoder, args.num_batch, shard_shape)
    log_str = (
        f'It will go through {dev_steps_per_epoch} steps to handle '
        f'{dev_dataset_size} data for development.')
    print_fn(log_str)

    val_dataset_size = tfds_builder.info.splits[val_split].num_examples
    val_steps_per_epoch = math.ceil(val_dataset_size / args.num_batch)
    val_dataset = tfds_builder.as_dataset(
        split=val_split, shuffle_files=False,
        decoders={'image': tensorflow_datasets.decode.SkipDecoding()})
    val_iter = create_val_iter(
        val_dataset, val_dataset_size,
        image_decoder, args.num_batch, shard_shape)
    log_str = (
        f'It will go through {val_steps_per_epoch} steps to handle '
        f'{val_dataset_size} data for validation.')
    print_fn(log_str)

    if args.data_augmentation == 'simple':
        trn_augmentation = None
    elif args.data_augmentation == 'colour':
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
    else:
        raise NotImplementedError(
            f'Unknown args.data_augmentation={args.data_augmentation}')

    # ----------------------------------------------------------------------- #
    # Model
    # ----------------------------------------------------------------------- #
    pixel_m = np.array([0.485, 0.456, 0.406])
    pixel_s = np.array([0.229, 0.224, 0.225])
    model = ResNet20x8()

    init_position = {
        'ext': model.init(
            jax.random.PRNGKey(args.seed),
            jnp.ones((1,) + input_shape))['params'],
        'cls': {
            'kernel': jax.random.normal(
                jax.random.PRNGKey(args.seed), (512, num_classes)),
            'bias': jnp.zeros((num_classes,))}}

    if args.ckpt:
        init_position = load(args.ckpt).position

    # ----------------------------------------------------------------------- #
    # Run
    # ----------------------------------------------------------------------- #
    def softmax(logits):
        # pylint: disable=redefined-outer-name
        """Computes softmax of logits."""
        return np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

    def compute_err(logits, labels):
        # pylint: disable=redefined-outer-name
        """Computes classification error."""
        return np.mean(np.not_equal(np.argmax(logits, axis=-1), labels))

    def compute_nll(logits, labels):
        # pylint: disable=redefined-outer-name
        """Computes categorical negative log-likelihood."""
        return np.mean(np.negative(
            np.log(softmax(logits)[np.arange(labels.shape[0]), labels])))

    def get_metrics(device_metrics):
        """Get metrics."""
        return jax.tree_util.tree_map(
            lambda *args: np.stack(args), *jax.device_get(
                jax.tree_util.tree_map(lambda x: x[0], device_metrics)))

    def forward_fn(params, inputs):
        # pylint: disable=redefined-outer-name
        """Computes categorical logits."""
        inputs = inputs / 255.0
        inputs = inputs - pixel_m[None, None, None]
        inputs = inputs / pixel_s[None, None, None]

        logits = model.apply({'params': params['ext']}, inputs)
        logits = logits @ params['cls']['kernel']
        logits = logits + params['cls']['bias'][None]

        return logits

    p_forward_fn = jax.pmap(forward_fn)

    def make_predictions(
            replicated_params,
            _steps_per_epoch, _iter, _dataset_size):
        # pylint: disable=redefined-outer-name
        """Returns logits and labels for val split."""
        logits = []
        labels = []
        for _ in range(_steps_per_epoch):
            _batch = next(_iter)
            logits.append(np.array(p_forward_fn(
                replicated_params, _batch['images']).reshape(-1, num_classes)))
            labels.append(np.array(_batch['labels'].reshape(-1)))
        return (
            np.concatenate(logits)[:_dataset_size],
            np.concatenate(labels)[:_dataset_size])

    def loss_fn(param, batch):
        # pylint: disable=redefined-outer-name
        """Computes cross-entropy loss."""
        logits = forward_fn(param, batch['images'])
        target = jax.nn.one_hot(batch['labels'], num_classes)

        cross_entropy_loss = jnp.negative(jnp.mean(
            jnp.sum(target * jax.nn.log_softmax(logits), axis=-1)))

        aux = OrderedDict({
            'cross_entropy_loss': cross_entropy_loss})

        return cross_entropy_loss, aux

    @partial(jax.pmap, axis_name='batch')
    def update_fn(state, batch, learning_rate):
        # pylint: disable=redefined-outer-name
        """Updates state."""
        aux, state = msgd.step(
            state=state,
            batch=batch,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            l2_regularizer=args.optim_l2,
            wd_regularizer=args.optim_wd*learning_rate/args.optim_lr,
            momentum=args.momentum,
            nesterov=args.nesterov,
            has_aux=True, axis_name='batch', grad_mask=None)
        aux[1]['lr'] = learning_rate
        return aux, state

    init_momentum = \
        jax.tree_util.tree_map(jnp.zeros_like, init_position)

    state = msgd.MSGDState(
        step=0, position=init_position, momentum=init_momentum)
    state = jax.device_put_replicated(state, jax.local_devices())

    ens_dev_ps = np.zeros((dev_dataset_size, num_classes))
    ens_dev_ls = np.zeros((dev_dataset_size, num_classes))
    ens_dev_ls_nlls = []

    ens_val_ps = np.zeros((val_dataset_size, num_classes))
    ens_val_ls = np.zeros((val_dataset_size, num_classes))
    ens_val_ls_nlls = []

    if args.save:
        sample_idx = 0 # pylint: disable=invalid-name
        save(
            os.path.join(args.save, f'{sample_idx:06d}'),
            jax.tree_util.tree_map(lambda e: e[0], state))

    for sample_idx in range(1, args.num_samples + 1):
        metrics = []
        for update_idx in range(1, args.num_updates + 1):

            batch = next(trn_iter)
            if trn_augmentation:
                batch['images'] = 255. * trn_augmentation(
                    jax.random.split(
                        jax.random.PRNGKey(update_idx), args.num_batch
                    ), batch['images'].reshape(-1, 224, 224, 3) / 255.
                ).reshape(batch['images'].shape)

            learning_rate = jax.device_put_replicated(
                args.optim_lr_min + (0.5 + 0.5 * np.cos(
                    (update_idx - 1) / args.num_updates * np.pi)
                ) * (args.optim_lr - args.optim_lr_min),
                jax.local_devices())
            aux, state = update_fn(state, batch, learning_rate)
            metrics.append(aux[1])

            if update_idx == 1 or update_idx % 5000 == 0:
                summarized = {
                    f'trn/{k}': float(v) for k, v in jax.tree_util.tree_map(
                        lambda e: e.mean(), get_metrics(metrics)).items()}

                if update_idx != 1:
                    metrics = []

                summarized['norm'] = float(jnp.sqrt(sum(
                    jnp.sum(e**2) for e in jax.tree_util.tree_leaves(
                        jax.tree_util.tree_map(
                            lambda e: e[0], state.position)))))

                logits, labels = make_predictions(
                    state.position,
                    dev_steps_per_epoch, dev_iter, dev_dataset_size)
                summarized['dev/err'] = compute_err(logits, labels)
                summarized['dev/nll'] = compute_nll(logits, labels)

                logits, labels = make_predictions(
                    state.position,
                    val_steps_per_epoch, val_iter, val_dataset_size)
                summarized['val/err'] = compute_err(logits, labels)
                summarized['val/nll'] = compute_nll(logits, labels)

                print_fn(
                    f'[Sample {sample_idx:6d}/{args.num_samples:6d}] '
                    f'[Update {update_idx:6d}/{args.num_updates:6d}] '
                    + ', '.join(
                        f'{k}: {v:.3e}' for k, v in summarized.items()))

                if args.use_wandb:
                    wandb.log(summarized)

                if jnp.isnan(summarized['trn/cross_entropy_loss']):
                    break

        if jnp.isnan(summarized['trn/cross_entropy_loss']):
            break

        summarized = {}

        summarized['norm'] = float(jnp.sqrt(sum(
            jnp.sum(e**2) for e in jax.tree_util.tree_leaves(
                jax.tree_util.tree_map(lambda e: e[0], state.position)))))

        logits, labels = make_predictions(
            state.position,
            dev_steps_per_epoch, dev_iter, dev_dataset_size)
        summarized['dev/err'] = compute_err(logits, labels)
        summarized['dev/nll'] = compute_nll(logits, labels)

        ens_dev_ps = (
            ens_dev_ps * (sample_idx - 1) + softmax(logits)) / sample_idx
        summarized['dev/ens_err'] = compute_err(np.log(ens_dev_ps), labels)
        summarized['dev/ens_nll'] = compute_nll(np.log(ens_dev_ps), labels)

        ens_dev_ls = (ens_dev_ls * (sample_idx - 1) + logits) / sample_idx
        ens_dev_ls_nlls.append(summarized['dev/nll'])
        summarized['dev/ens_amb'] = \
            np.mean(ens_dev_ls_nlls) - compute_nll(ens_dev_ls, labels)

        logits, labels = make_predictions(
            state.position,
            val_steps_per_epoch, val_iter, val_dataset_size)
        summarized['val/err'] = compute_err(logits, labels)
        summarized['val/nll'] = compute_nll(logits, labels)

        ens_val_ps = (
            ens_val_ps * (sample_idx - 1) + softmax(logits)) / sample_idx
        summarized['val/ens_err'] = compute_err(np.log(ens_val_ps), labels)
        summarized['val/ens_nll'] = compute_nll(np.log(ens_val_ps), labels)

        ens_val_ls = (ens_val_ls * (sample_idx - 1) + logits) / sample_idx
        ens_val_ls_nlls.append(summarized['val/nll'])
        summarized['val/ens_amb'] = \
            np.mean(ens_val_ls_nlls) - compute_nll(ens_val_ls, labels)

        print_fn(
            f'[Sample {sample_idx:6d}/{args.num_samples:6d}] '
            + ', '.join(f'{k}: {v:.3e}' for k, v in summarized.items()))

        if args.save:
            save(
                os.path.join(args.save, f'{sample_idx:06d}'),
                jax.tree_util.tree_map(lambda e: e[0], state))
