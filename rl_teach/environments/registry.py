from .cartpole import CartPoleEnv
from .mountaincar import MountainCarEnv

ENV_REGISTRY = {
    "cartpole": CartPoleEnv,
    "mountaincar": MountainCarEnv,
}


def make_env(name, max_steps=200, num_digitized=6):
    if name not in ENV_REGISTRY:
        raise ValueError(
            "未知环境 '{}'. 可选: {}".format(name, list(ENV_REGISTRY.keys())))
    return ENV_REGISTRY[name](max_steps=max_steps, num_digitized=num_digitized)
