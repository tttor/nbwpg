import numpy as np, gym

def make_single_env(env_id, seed):
    env = gym.make(env_id)
    env.seed(seed)
    if env.action_space is not None:
        env.action_space.np_random.seed(seed) # https://github.com/openai/gym/issues/681
    return env
