# nbwpg: experiment

# scheme
There are four experiment schemes, namely
* `optexact-mesh`: exact optimization,
* `optsampling-mesh`: sampling-based optimization,
* `envprop-mesh`: environment properties (including the optimization landscape),
* `biasgradexact-mesh`: timestep-wise decomposition of the exact gradients of the bias.

# setup
* Follow `../main/README.md`
* The `logdir` item in every config `cfg/foo.yaml` (under each scheme)
  is relative to the home directory.

