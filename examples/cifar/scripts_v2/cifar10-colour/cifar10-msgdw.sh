for SEED in 0 1 2 3 4 5 6 7 8 9 ; \
  do python examples/cifar/run_msgd.py \
    --data_name cifar10 \
    --data_augmentation colour \
    --optim_lr 0.03 \
    --optim_lr_min 0.0 \
    --optim_wd 0.0003 \
    --momentum 0.9 \
    --seed $SEED \
    --save examples/cifar/save_v2/cifar10-colour/msgdw/$SEED ; done