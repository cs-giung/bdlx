for PT in 1.0 0.1 0.01 0.001 0.0001 ; \
  do for SEED in 0 1 2 3 ; \
    do python examples/cifar/run_sgld.py \
      --data_name cifar10 \
      --data_augmentation simple \
      --step_size 0.00001 \
      --step_size_min 0.0 \
      --posterior_temperature $PT \
      --prior_variance 0.1 \
      --seed $SEED \
      --save examples/cifar/save/cifar10-simple/sgld-cyclical-tempering/pt$PT-$SEED ; done ; done