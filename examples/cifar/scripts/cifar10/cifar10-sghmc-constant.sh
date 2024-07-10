for SEED in 0 1 ; \
  do python examples/cifar/run_sghmc.py \
    --data_name cifar10 \
    --step_size 0.0003 \
    --posterior_temperature 1.0 \
    --prior_variance 0.05 \
    --friction 100.0 \
    --seed $SEED \
    --save examples/cifar/save/cifar10/sghmc-constant/$SEED ; done