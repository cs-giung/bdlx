for SEED in 0 1 2 3 4 5 6 7 8 9 ; \
  do python examples/cifar/run_adam.py \
    --data_name cifar10 \
    --optim_lr 0.001 \
    --optim_lr_min 0.0 \
    --optim_l2 0.01 \
    --momentum_mu 0.9 \
    --momentum_nu 0.99999 \
    --seed $SEED \
    --save examples/cifar/save/cifar10/adam/$SEED ; done