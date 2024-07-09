for SEED in 0 1 2 3 4 5 6 7 8 9 ; \
  do python examples/cifar/run_sghmc.py \
    --data_name cifar10 \
    --step_size 0.0003 \
    --step_size_min 0.0 \
    --posterior_temperature 1.0 \
    --prior_variance 0.025 \
    --friction 100.0 \
    --seed $SEED \
    --save examples/cifar/save/cifar10/sghmc-cyclical/$SEED ; done