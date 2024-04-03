import jax.numpy as jnp
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import save_to_zip_file, load_from_zip_file


def to_simplex(a: jnp.ndarray) -> jnp.ndarray:
    tmp = a - a.min()
    return tmp / tmp.sum()


def cosine_sim(a: jnp.ndarray, b: jnp.ndarray) -> float:
    return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))


def manhattan_dist(a: jnp.ndarray, b: jnp.ndarray) -> float:
    return float(jnp.abs(a - b).sum())


def save_policy(model: BasePolicy, path: str):
    pytorch_variables = {"policy": model}
    params_to_save = {"policy": model.state_dict()}
    save_to_zip_file(path, params=params_to_save, pytorch_variables=pytorch_variables)


def load_policy(path: str) -> BasePolicy:
    _, params, pytorch_variables = load_from_zip_file(path)
    assert pytorch_variables is not None, "No pytorch variables found in model file"
    assert params is not None, "No parameters found in model file"
    try:
        policy = pytorch_variables["policy"]
        policy.load_state_dict(params["policy"])
        return policy
    except:
        raise ValueError("Failed to load policy")
