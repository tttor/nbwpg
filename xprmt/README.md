# nbwpg: experiment

# Scheme
There are four experiment schemes, namely
* `optexact-mesh`: exact optimization,
* `optsampling-mesh`: sampling-based optimization,
* `envprop-mesh`: environment properties (including the optimization landscape),
* `biasgradexact-mesh`: timestep-wise decomposition of the exact gradients of the bias.

# Setup
* Follow `../main/README.md`
* The `logdir` item in every config `cfg/foo.yaml` (under each scheme)
  is relative to the home directory.

# Environment name to config mappings
* Env-A1: `../gym-env/gym-symbol/gym_symbol/envs/config/example_10_1_2_v1.yaml`
* Env-A2: `../gym-env/gym-symbol/gym_symbol/envs/config/tor_20210307_v1.yaml`
* Env-A3: `../gym-env/gym-symbol/gym_symbol/envs/config/gridnav_2_v0.yaml`
* Env-B1: `../gym-env/gym-symbol/gym_symbol/envs/config/tor_20201121a_v1.yaml`
* Env-B2: `../gym-env/gym-symbol/gym_symbol/envs/config/hordijk_example_v4.yaml`
* Env-B3: `../gym-env/gym-symbol/gym_symbol/envs/config/nchain_mod_v1.yaml`
