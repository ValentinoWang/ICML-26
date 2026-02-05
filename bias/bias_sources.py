from __future__ import annotations

import numpy as np


def ridge_estimator(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    d = X.shape[1]
    lhs = X.T @ X + alpha * np.eye(d)
    rhs = X.T @ y
    return np.linalg.solve(lhs, rhs)


def bayesian_map_estimator(
    X: np.ndarray,
    y: np.ndarray,
    *,
    mu_prior: np.ndarray | float,
    sigma_prior: np.ndarray | float,
    noise_var: float,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    d = X.shape[1]
    mu_prior = np.asarray(mu_prior, dtype=float)
    if mu_prior.ndim == 0:
        mu_prior = np.full(d, float(mu_prior))
    sigma_prior = np.asarray(sigma_prior, dtype=float)
    if sigma_prior.ndim == 0:
        prior_prec = np.eye(d) / (float(sigma_prior) ** 2)
        prior_rhs = mu_prior / (float(sigma_prior) ** 2)
    else:
        prior_prec = np.diag(1.0 / (sigma_prior**2))
        prior_rhs = mu_prior / (sigma_prior**2)
    lhs = (X.T @ X) / noise_var + prior_prec
    rhs = (X.T @ y) / noise_var + prior_rhs
    return np.linalg.solve(lhs, rhs)


def const_bias_estimator(theta_hat: np.ndarray, bias_vec: np.ndarray) -> np.ndarray:
    return np.asarray(theta_hat, dtype=float) + np.asarray(bias_vec, dtype=float)
