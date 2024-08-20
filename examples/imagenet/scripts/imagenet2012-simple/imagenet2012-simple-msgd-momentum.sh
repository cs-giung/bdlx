for SEED in 0 1 2 3 ; \
  do python examples/imagenet2012/run_msgd.py \
    --data_name imagenet2012 \
    --data_augmentation simple \
    --optim_lr 0.1 \
    --optim_lr_min 0.0 \
    --optim_l2 0.0001 \
    --momentum 0.9 \
    --nesterov false \
    --seed $SEED \
    --save examples/imagenet2012/save/imagenet2012-simple/msgd-momentum/$SEED ; done