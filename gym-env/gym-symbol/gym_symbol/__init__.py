from gym.envs.registration import register

# gridnav: square ##############################################################
register(
    id='GridNav_2-v0',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'gridnav_2_v0.yaml'}
)

register(
    id='GridNav_2-v1',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'gridnav_2_v1.yaml'}
)

register(
    id='GridNav_3-v0',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'gridnav_3_v0.yaml'}
)

register(
    id='GridNav_3-v1',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'gridnav_3_v1.yaml'}
)

# nchain modified ##############################################################
# gym.error.Error: Cannot re-register id: NChain-v0
register(
    id='NChain_mod-v0',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'nchain_mod_v0.yaml'}
)

register(
    id='NChain_mod-v1',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'nchain_mod_v1.yaml'}
)

# tor ##########################################################################
register(
    id='Tor_20201121a-v0',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'tor_20201121a.yaml'}
)

register(
    id='Tor_20201121a-v1',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'tor_20201121a_v1.yaml'}
)

register(
    id='hordijk_example-v0',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'hordijk_example_v0.yaml'}
)

register(
    id='Hordijk_example-v3',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'hordijk_example_v3.yaml'}
)

register(
    id='Hordijk_example-v4',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'hordijk_example_v4.yaml'}
)

register(
    id='Tor_20210306-v0',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'tor_20210306_v0.yaml'}
)

register(
    id='Tor_20210306-v1',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'tor_20210306_v1.yaml'}
)

register(
    id='Tor_20210307-v0',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'tor_20210307_v0.yaml'}
)

register(
    id='Tor_20210307-v1',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'tor_20210307_v1.yaml'}
)

# feinberg_2002_hmdp ###########################################################
register(
    id='Example_3_1-v0',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'example_3_1.yaml'}
)

register(
    id='Example_3_3-v0',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'example_3_3.yaml'}
)

register(
    id='Example_8_1-v0',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'example_8_1.yaml'}
)

# puterman_1994_mdp ############################################################
register(
    id='Example_10_1_1-v0',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'example_10_1_1.yaml'}
)

register(
    id='Example_10_1_2-v0',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'example_10_1_2.yaml'}
)

register(
    id='Example_10_1_2-v1',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'example_10_1_2_v1.yaml'}
)

register(
    id='Example_10_2_2-v0',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'example_10_2_2.yaml'}
)

register(
    id='Problem_10_7-v0',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'problem_10_7.yaml'}
)

register(
    id='Problem_10_9-v0',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'problem_10_9.yaml'}
)

register(
    id='Problem_6_64-v0',
    entry_point='gym_symbol.envs:SymbolicRepresentation',
    kwargs={'cfg_fname': 'problem_6_64.yaml'}
)
