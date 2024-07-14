for SEED in 0 ; \
  do python examples/ivon/run_ivon.py \
    --data_name imagenet2012 \
    --data_augmentation simple \
    --ess_factor 1.0 \
    --hess_init 1.0 \
    --momentum_mu 0.9 \
    --momentum_nu 0.99999 \
    --optim_lr 0.3 \
    --optim_lr_min 0.0 \
    --optim_l2 0.0001 \
    --seed $SEED \
    --save examples/imagenet2012/save/imagenet2012-simple/ivon/$SEED ; done