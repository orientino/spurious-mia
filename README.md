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
