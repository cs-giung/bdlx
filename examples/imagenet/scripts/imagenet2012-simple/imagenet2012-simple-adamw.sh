for SEED in 0 1 2 3 ; \
  do python examples/imagenet2012/run_adam.py \
    --data_name imagenet2012 \
    --data_augmentation simple \
    --optim_lr 0.003 \
    --optim_lr_min 0.0 \
    --optim_wd 0.0001 \
    --momentum_nu 0.9999 \
    --seed $SEED \
    --save examples/imagenet2012/save/imagenet2012-simple/adamw/$SEED ; done