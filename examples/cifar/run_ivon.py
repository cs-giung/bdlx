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
import wandb

from bdlx.optim import ivon
from bdlx.tree_util import load, save
from examples.cifar import image_processing
from examples.cifar.default import get_args, str2bool
from examples.cifar.model import ResNet20


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument(
        '--ckpt', default=None, type=str,
        help='initialize position from *.ckpt if specified (default: None)')

    parser.add_argument(
        '--data_root', default='./examples/cifar/data/', type=str,
        help='root directory containing dataset files')
    parser.add_argument(
        '--data_name', default='cifar10', type=str,
        help='dataset name (default: cifar10)')
    parser.add_argument(
        '--data_augmentation', default='none', type=str,
        help='apply data augmentation during training (default: none)')

    parser.add_argument(
        '--num_samples', default=1, type=int,
        help='the number of samples (default: 1)')
    parser.add_argument(
        '--num_updates', default=50000, type=int,
        help='the number of updates for each sample (default: 50000)')
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
        '--hess_init', default=1.0, type=float,
        help='it determines initial hessian estimate (default: 1.0)')
    parser.add_argument(
        '--ess_factor', default=1.0, type=float,
        help='it determines effective sample size (default: 1.0)')
    parser.add_argument(
        '--momentum_mu', default=0.9, type=float,
        help='momentum coefficient (default: 0.9)')
    parser.add_argument(
        '--momentum_nu', default=0.99999, type=float,
        help='momentum coefficient (default: 0.99999)')

    parser.add_argument(
        '--use_wandb', default=False, type=str2bool,
        help='use wandb if specified (default: False)')

    args, print_fn, time_stamp = get_args(
        parser, exist_ok=True, dot_log_file=False,
        libraries=(flax, jax, jaxlib))

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
    input_shape = (32, 32, 3)
    num_classes = {'cifar10': 10, 'cifar100': 100}[args.data_name]

    trn_images = np.load(os.path.join(
        args.data_root, args.data_name, 'train_images.npy'))[:40960]
    trn_labels = np.load(os.path.join(
        args.data_root, args.data_name, 'train_labels.npy'))[:40960]
    trn_dataset_size = trn_images.shape[0]
    trn_steps_per_epoch = math.floor(trn_dataset_size / args.num_batch)

    val_images = np.load(os.path.join(
        args.data_root, args.data_name, 'train_images.npy'))[40960:]
    val_labels = np.load(os.path.join(
        args.data_root, args.data_name, 'train_labels.npy'))[40960:]

    if args.data_augmentation == 'none':
        trn_augmentation = None
    elif args.data_augmentation == 'simple':
        trn_augmentation = jax.jit(jax.vmap(image_processing.TransformChain([
            image_processing.RandomCropTransform(size=32, padding=4),
            image_processing.RandomHFlipTransform(prob=0.5)])))
    elif args.data_augmentation == 'colour':
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
    else:
        raise NotImplementedError(
            f'Unknown args.data_augmentation={args.data_augmentation}')

    # ----------------------------------------------------------------------- #
    # Model
    # ----------------------------------------------------------------------- #
    pixel_m = np.array([0.49, 0.48, 0.44])
    pixel_s = np.array([0.2, 0.2, 0.2])
    model = ResNet20()

    init_position = {
        'ext': model.init(
            jax.random.PRNGKey(args.seed),
            jnp.ones((1,) + input_shape))['params'],
        'cls': {
            'kernel': jax.random.normal(
                jax.random.PRNGKey(args.seed), (64, num_classes)),
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

    def make_predictions(replicated_params, images):
        # pylint: disable=redefined-outer-name
        """Returns logits and labels for val split."""
        dataset_size = images.shape[0]
        steps_per_epoch = math.ceil(dataset_size / args.num_batch)

        images = np.concatenate([images, np.zeros((
            args.num_batch * steps_per_epoch - dataset_size,
            *images.shape[1:]), dtype=images.dtype)])
        _queue = np.arange(images.shape[0]).reshape(-1, args.num_batch)

        logits = np.concatenate([
            jax.device_put(p_forward_fn(
                replicated_params, images[batch_index].reshape(
                    shard_shape + input_shape)).reshape(args.num_batch, -1),
                jax.devices('cpu')[0]) for batch_index in _queue])
        logits = logits[:dataset_size]

        return logits

    def loss_fn(param, batch):
        # pylint: disable=redefined-outer-name
        """Computes cross-entropy loss."""
        logits = forward_fn(param, batch['inputs'])
        target = batch['target']

        cross_entropy_loss = jnp.negative(jnp.mean(
            jnp.sum(target * jax.nn.log_softmax(logits), axis=-1)))

        aux = OrderedDict({
            'cross_entropy_loss': cross_entropy_loss})

        return cross_entropy_loss, aux

    @partial(jax.pmap, axis_name='batch')
    def update_fn(state, batch, learning_rate):
        # pylint: disable=redefined-outer-name
        """Updates state."""
        aux, state = ivon.step(
            state=state,
            batch=batch,
            loss_fn=loss_fn,
            effective_sample_size=args.ess_factor*trn_dataset_size,
            learning_rate=learning_rate,
            l2_regularizer=args.optim_l2,
            wd_regularizer=args.optim_wd,
            momentums=(args.momentum_mu, args.momentum_nu),
            has_aux=True, axis_name='batch', grad_mask=None)
        aux[1]['lr'] = learning_rate
        return aux, state

    init_momentum_mu = \
        jax.tree_util.tree_map(
            partial(jnp.full_like, fill_value=0.0),
            init_position)
    init_momentum_nu = \
        jax.tree_util.tree_map(
            partial(jnp.full_like, fill_value=args.hess_init),
            init_position)

    state = ivon.IVONState(
        step=0, rng_key=jax.random.PRNGKey(args.seed), position=init_position,
        momentum_mu=init_momentum_mu, momentum_nu=init_momentum_nu)
    state = jax.device_put_replicated(state, jax.local_devices())

    batch_rng = jax.random.PRNGKey(args.seed)
    batch_queue = np.asarray(
        jax.random.permutation(batch_rng, trn_dataset_size))

    ens_trn_ps = np.zeros((trn_images.shape[0], num_classes))
    ens_trn_ls = np.zeros((trn_images.shape[0], num_classes))
    ens_trn_ls_nlls = []

    ens_val_ps = np.zeros((val_images.shape[0], num_classes))
    ens_val_ls = np.zeros((val_images.shape[0], num_classes))
    ens_val_ls_nlls = []

    if args.save:
        sample_idx = 0 # pylint: disable=invalid-name
        save(
            os.path.join(args.save, f'{sample_idx:06d}'),
            jax.tree_util.tree_map(lambda e: e[0], state))

    for sample_idx in range(1, args.num_samples + 1):
        metrics = []
        for update_idx in range(1, args.num_updates + 1):

            if batch_queue.shape[0] <= args.num_batch:
                batch_rng = jax.random.split(batch_rng)[0]
                batch_queue = np.concatenate((batch_queue,
                    jax.random.permutation(batch_rng, trn_dataset_size)))
            batch_index = batch_queue[:args.num_batch]
            batch_queue = batch_queue[args.num_batch:]

            batch = {
                'inputs': trn_images[batch_index],
                'target': jax.nn.one_hot(trn_labels[batch_index], num_classes)}
            if trn_augmentation:
                batch['inputs'] = (trn_augmentation(
                    jax.random.split(
                        jax.random.PRNGKey(update_idx), args.num_batch
                    ), batch['inputs'] / 255.) * 255.).astype(trn_images.dtype)
            batch = jax.tree_util.tree_map(
                lambda e: e.reshape(shard_shape + e.shape[1:]), batch)

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

                logits = make_predictions(state.position, trn_images)
                summarized['trn/err'] = compute_err(logits, trn_labels)
                summarized['trn/nll'] = compute_nll(logits, trn_labels)

                logits = make_predictions(state.position, val_images)
                summarized['val/err'] = compute_err(logits, val_labels)
                summarized['val/nll'] = compute_nll(logits, val_labels)

                print_fn(
                    f'[Sample {sample_idx:6d}/{args.num_samples:6d}] '
                    f'[Update {update_idx:6d}/{args.num_updates:6d}] '
                    + ', '.join(
                        f'{k}: {v:.3e}' for k, v in summarized.items()))

                if args.use_wandb:
                    wandb.log(summarized)

                if jnp.isnan(summarized['trn/cross_entropy_loss']):
                    break

        summarized = {}

        summarized['norm'] = float(jnp.sqrt(sum(
            jnp.sum(e**2) for e in jax.tree_util.tree_leaves(
                jax.tree_util.tree_map(lambda e: e[0], state.position)))))

        logits = make_predictions(state.position, trn_images)
        summarized['trn/err'] = compute_err(logits, trn_labels)
        summarized['trn/nll'] = compute_nll(logits, trn_labels)

        ens_trn_ps = (
            ens_trn_ps * (sample_idx - 1) + softmax(logits)) / sample_idx
        summarized['trn/ens_err'] = compute_err(np.log(ens_trn_ps), trn_labels)
        summarized['trn/ens_nll'] = compute_nll(np.log(ens_trn_ps), trn_labels)

        ens_trn_ls = (ens_trn_ls * (sample_idx - 1) + logits) / sample_idx
        ens_trn_ls_nlls.append(summarized['trn/nll'])
        summarized['trn/ens_amb'] = \
            np.mean(ens_trn_ls_nlls) - compute_nll(ens_trn_ls, trn_labels)

        logits = make_predictions(state.position, val_images)
        summarized['val/err'] = compute_err(logits, val_labels)
        summarized['val/nll'] = compute_nll(logits, val_labels)

        ens_val_ps = (
            ens_val_ps * (sample_idx - 1) + softmax(logits)) / sample_idx
        summarized['val/ens_err'] = compute_err(np.log(ens_val_ps), val_labels)
        summarized['val/ens_nll'] = compute_nll(np.log(ens_val_ps), val_labels)

        ens_val_ls = (ens_val_ls * (sample_idx - 1) + logits) / sample_idx
        ens_val_ls_nlls.append(summarized['val/nll'])
        summarized['val/ens_amb'] = \
            np.mean(ens_val_ls_nlls) - compute_nll(ens_val_ls, val_labels)

        print_fn(
            f'[Sample {sample_idx:6d}/{args.num_samples:6d}] '
            + ', '.join(f'{k}: {v:.3e}' for k, v in summarized.items()))

        if args.save:
            save(
                os.path.join(args.save, f'{sample_idx:06d}'),
                jax.tree_util.tree_map(lambda e: e[0], state))
