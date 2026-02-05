import numpy as np


def set_seed(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_bias_vector(dim: int, target_norm: float) -> np.ndarray:
    vec = np.ones(dim)
    return target_norm * vec / np.linalg.norm(vec)


def sample_candidates(
    rng: np.random.Generator,
    theta_true: np.ndarray,
    bias: np.ndarray,
    n: int,
    noise_std: float,
) -> np.ndarray:
    """
    Generate a candidate set with systematic bias plus Gaussian noise.
    """
    shift = theta_true + bias
    return shift + noise_std * rng.normal(size=(n, theta_true.shape[0]))


def sample_clean_reference(
    rng: np.random.Generator,
    theta_true: np.ndarray,
    n: int,
    noise_std: float,
) -> np.ndarray:
    """
    Small unbiased batch used to approximate theta_good.
    """
    samples = theta_true + noise_std * rng.normal(size=(n, theta_true.shape[0]))
    return samples.mean(axis=0)
