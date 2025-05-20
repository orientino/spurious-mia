## Getting started
Prepare the environment
```
conda create -f env.yml -n spurious
python3 setup.py develop
```

## Run
Run the LiRA attack on the Waterbirds dataset:
```
./gpu.sh
```
This will generate a file with the attack result, revealing the spurious privacy leakage phenomenon.

## Extras
`train.py` has all the configuration to train spurious robust models with differential privacy flags. `train_dfr.py` takes a model trained with ERM and train an DFR model. 
