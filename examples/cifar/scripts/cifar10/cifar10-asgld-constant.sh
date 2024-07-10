for SEED in 0 1 ; \
  do python examples/cifar/run_asgld.py \
    --data_name cifar10 \
    --step_size 0.00001 \
    --posterior_temperature 1.0 \
    --prior_variance 0.2 \
    --seed $SEED \
    --save examples/cifar/save/cifar10/asgld-constant/$SEED ; done