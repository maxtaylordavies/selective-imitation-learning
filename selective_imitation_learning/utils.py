import io

from PIL import Image
import jax.numpy as jnp
import matplotlib.pyplot as plt
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import save_to_zip_file, load_from_zip_file


def is_power(n, p):
    root = round(n ** (1 / p))
    return root**p == n


def to_range(a: jnp.ndarray, low, high) -> jnp.ndarray:
    a = (a - a.min()) / (a.max() - a.min())
    return a * (high - low) + low


def to_simplex(a: jnp.ndarray) -> jnp.ndarray:
    a = a - a.min() + 1e-8
    return a / a.sum()


def cosine_sim(a: jnp.ndarray, b: jnp.ndarray) -> float:
    return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))


def manhattan_dist(a: jnp.ndarray, b: jnp.ndarray):
    return jnp.abs(a - b).sum()


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


def fig_to_pil_image(fig, format="png", close=True):
    buf = io.BytesIO()
    fig.savefig(buf, format=format)
    buf.seek(0)
    if close:
        plt.close(fig)
    return Image.open(buf)


def images_to_gif(imgs, path, duration=100):
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)
