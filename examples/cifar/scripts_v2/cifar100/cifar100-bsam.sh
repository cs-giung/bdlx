for SEED in 0 1 2 3 4 5 6 7 8 9 ; \
  do python examples/cifar/run_bsam.py \
    --data_name cifar100 \
    --data_augmentation none \
    --optim_lr 0.3 \
    --optim_lr_min 0.0 \
    --optim_l2 0.001 \
    --momentum_mu 0.9 \
    --momentum_nu 0.9999 \
    --ess_factor 1.0 \
    --eps 0.1 \
    --radius 0.03 \
    --seed $SEED \
    --save examples/cifar/save_v2/cifar100/bsam/$SEED ; done