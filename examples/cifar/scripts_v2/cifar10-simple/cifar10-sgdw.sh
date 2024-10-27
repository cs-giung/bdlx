for SEED in 0 1 2 3 4 5 6 7 8 9 ; \
  do python examples/cifar/run_sgd.py \
    --data_name cifar10 \
    --data_augmentation simple \
    --optim_lr 0.3 \
    --optim_lr_min 0.0 \
    --optim_wd 0.0003 \
    --seed $SEED \
    --save examples/cifar/save_v2/cifar10-simple/sgdw/$SEED ; done