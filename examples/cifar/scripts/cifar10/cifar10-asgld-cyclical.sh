for SEED in 0 1 ; \
  do python examples/cifar/run_asgld.py \
    --data_name cifar10 \
    --step_size 0.000003 \
    --step_size_min 0.0 \
    --posterior_temperature 1.0 \
    --prior_variance 0.05 \
    --seed $SEED \
    --save examples/cifar/save/cifar10/asgld-cyclical/$SEED ; done