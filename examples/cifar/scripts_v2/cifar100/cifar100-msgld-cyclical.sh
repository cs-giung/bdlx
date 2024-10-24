for SEED in 0 1 2 3 4 5 6 7 8 9 ; \
  do python examples/cifar/run_msgld.py \
    --data_name cifar100 \
    --data_augmentation none \
    --step_size 0.000003 \
    --step_size_min 0.0 \
    --posterior_temperature 1.0 \
    --prior_variance 0.1 \
    --bias 1.0 \
    --smoothing 0.9 \
    --seed $SEED \
    --save examples/cifar/save_v2/cifar100/msgld-cyclical/$SEED ; done
