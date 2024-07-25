for SEED in 0 1 2 3 4 5 6 7 8 9 ; \
  do python examples/cifar/run_sgd.py \
    --data_name cifar100 \
    --optim_lr 0.1 \
    --optim_lr_min 0.0 \
    --optim_l2 0.003 \
    --seed $SEED \
    --save examples/cifar/save/cifar100/sgd/$SEED ; done