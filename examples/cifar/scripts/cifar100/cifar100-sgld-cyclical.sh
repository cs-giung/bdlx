for SEED in 0 1 2 3 4 5 6 7 8 9 ; \
  do python examples/cifar/run_sgld.py \
    --data_name cifar100 \
    --step_size 0.00001 \
    --step_size_min 0.0 \
    --posterior_temperature 1.0 \
    --prior_variance 0.2 \
    --seed $SEED \
    --save examples/cifar/save/cifar100/sgld-cyclical/$SEED ; done