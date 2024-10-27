for SEED in 0 1 2 3 4 5 6 7 8 9 ; \
  do python examples/cifar/run_msgd.py \
    --data_name cifar100 \
    --data_augmentation colour \
    --optim_lr 0.3 \
    --optim_lr_min 0.0 \
    --optim_l2 0.0003 \
    --momentum 0.9 \
    --seed $SEED \
    --save examples/cifar/save_v2/cifar100-colour/msgd/$SEED ; done