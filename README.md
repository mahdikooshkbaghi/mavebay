[![Build](https://github.com/mahdikooshkbaghi/mavebay/actions/workflows/main.yaml/badge.svg)](https://github.com/mahdikooshkbaghi/mavebay/actions/workflows/main.yaml)

# Installation

One can install the package directly from the `github` repository or install it from the local cloned version.

The base installation `"mavebay"` avoids additional packages such as `matplotlib`, `arviz`, `logomaker` and etc.

The full installation which is suitable to run demos is also provided `"mavebay[examples]"`.

## Installing from `github`

1. Create the virtual environment.
2. Install the base or full `mavebay` package.

```bash
python -m venv test_mavebay
source test_mavebay/bin/activate
pip install git+https://github.com/mahdikooshkbaghi/mavebay "mavebay[examples]"
```

## Installing from the cloned repository 

1. Clone the repo.
2. Create the virtual environment.
3. Install the base or full `mavebay` package.

```bash
git clone git@github.com:mahdikooshkbaghi/mavebay.git
cd mavebay
python -m venv test_mavebay
source test_mavebay/bin/activate
pip install . "mavebay"
# OR
pip install . "mavebay[examples]"
```

# Demos

The global epistasis (GE) measurement process example script is provided in the `example` folder.
The following command can be used to run demos

```bash
python global_epistasis_demo.py -n [NUM_SAMPLES]        \
                                -lr [LEARNING_RATE]     \
                                -m [METHOD]             \
                                -ds [DATA]              \
                                -k [INTERACTION_ORDER]  \
                                -d [DEVICE]             \
                                -i [INIT_LOC_FN]        \
                                -p [PROGRESS_BAR]

```
All the arguments has some default values which are provided in the script.

- `NUM_SAMPLES`: number of samples for `mcmc` or number of steps in the `svi`.
- `LEARNING_RATE`: the learning rate for the optimizer in `svi` method.
- `METHOD`: method of inference: `mcmc` or `svi`
- `DATA`: dataset to use for the inference. Descriptions of the datasets are given in the MAVE-NN manuscript.
    - `abeta`: DMS data for Aβ (default dataset). 
    - `tdp43`: DMS data for TDP-43.
    - `mpsa`: MPSA data for 5' splicing sites.
- `INTERACTION_ORDER`: `k=1 (default)` corresponds to the additive GP map, `k=2` pairwise GP map and so on.
- `DEVICE`: `cpu (default)` or `gpu`.
- `INIT_LOC_FN`: initialization for the `svi` sampling. Default it `feasible`. Others can be assigned based on the `numpyro` documentation.
- `PROGRESS_BAR`: enable (default) or disable the progress bar of the inference. 


# TODO list for MAVEBAY
Bayesian Version of MAVENN1.0

- [x] The SVI is working both on the abeta and TDP-43 additive inferences.
- [x] The MCMC on the abeta is working only on small samples.
- [ ] Need to figure out the batch and plate. 
- [x] Need to modify the `setup.py` to have additional python requirements for the example folder. Check the numpyro github repo for hint.
- [ ] Implementing the skewed-T noise model similar to one we have in MAVENN.
- [x] K-th order interaction GP map implementation.
    - [x] The K-th=1: which is practically an additive model and it is working.
    - [x] The K-th=2: which is practically pairwise works on mpsa data.
- [ ] Measurement process agnostic (MPA) implementation.
- [x] Need to add different initialization strategy for SVI sampling with default being `init_to_feasible`
- [ ] Saving the model:
    - [ ] SVI
    - [ ] MCMC
- [x] Make the ppc smooth.
    - [x] As I suspected the number of samples from posteriors were not enough to make the phi_to_yhat smooth. Increasing that fixed the issue.
- [ ] Put the MAVENN heatmap and pairwise to utils function
- [x] Information metrics calculation.
