import math
import warnings
import numpy as np


def sample_GBM_log_paths(N: int, T: float, M: int, r: float, 
                     sigma: float, S_0: float, rng: np.random.Generator) -> np.ndarray:
    """
    Parameters
    ----------
    N : Number of GBM paths
    T : Maturity
    M : Number of discretization steps
    r : risk-free rate
    sigma : volatility term
    S_0 : initial price of the underlying
    rng : np.random.Generator that allows to fix the random values

    Returns
    -------
    a numpy array (N, M) of geometric brownian motion log paths sampled exactly
    """

    dt = T/M
    Z = rng.standard_normal((N, M))
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.log(S_0) + np.cumsum(increments, axis=1)
    return log_paths


def discounted_geometric_asian_payoff(log_paths: np.ndarray, strike: float, r: float, T: float) -> np.ndarray:
    """
    Parameters
    ----------
    log_paths : numpy array (N, M) of log GBM paths, column m-1 corresponds to time t_m
    strike : strike price K of the option
    r : risk-free rate
    T : Maturity (payment date of the option)

    Returns
    -------
    a numpy array (N,) of discounted payoffs C_i = exp(-r T) * max(G - K, 0)
    of the geometric Asian call
    """
    G = np.exp(np.mean(log_paths, axis=1))
    return np.exp(-r*T)*np.maximum(G - strike, 0.0)


def discounted_arithmetic_asian_payoff(log_paths: np.ndarray, strike: float, r: float, T: float) -> np.ndarray:
    """
    Parameters
    ----------
    log_paths : numpy array (N, M) of log GBM paths, column m-1 corresponds to time t_m
    strike : strike price K of the option
    r : risk-free rate
    T : Maturity (payment date of the option)

    Returns
    -------
    a numpy array (N,) of discounted payoffs A_i = exp(-r T) * max(S_bar - K, 0)
    of the arithmetic Asian call, where S_bar = (S_{t_1} + ... + S_{t_M}) / M
    """
    paths = np.exp(log_paths)
    paths_arithmetic_means = paths.mean(axis=1)
    return np.exp(-r*T)*np.maximum(paths_arithmetic_means - strike, 0.0)


def geometric_mean_lognormal_params(T: float, M: int, r: float, sigma: float,
                                    S_0: float) -> tuple[float, float]:
    """
    Parameters
    ----------
    T : Maturity
    M : Number of discretization steps
    r : risk-free rate
    sigma : volatility term
    S_0 : initial price of the underlying

    Returns
    -------
    (nu_G, tau_G) such that log G ~ N(nu_G, tau_G^2) for the geometric mean
    G = (S_{t_1} * ... * S_{t_M})^(1/M) on the grid t_m = m * dt, dt = T / M:
        nu_G    = log S_0 + (r - sigma^2 / 2) * dt * (M + 1) / 2
        tau_G^2 = sigma^2 * dt * (M + 1) * (2M + 1) / (6M)
    """
    dt = T / M
    nu_G = np.log(S_0) + (r - 0.5 * sigma**2) * dt * (M + 1) / 2
    tau_G = sigma * np.sqrt(dt * (M + 1) * (2 * M + 1) / (6 * M))
    return nu_G, tau_G


def discounted_geometric_asian_exact_mean(T: float, M: int, r: float, sigma: float,
                               S_0: float, strike: float) -> float:
    """
    Parameters
    ----------
    T : Maturity
    M : Number of discretization steps
    r : risk-free rate
    sigma : volatility term
    S_0 : initial price of the underlying
    strike : strike price K of the option

    Returns
    -------
    mu_C : exact mean E[C] = E[exp(-r T) * max(G - K, 0)] of the discounted
        geometric Asian call payoff,
        mu_C = exp(-r T) * (exp(nu_G + tau_G^2 / 2) * Phi(d_1) - K * Phi(d_2)),
        d_1 = (nu_G - log K + tau_G^2) / tau_G,  d_2 = d_1 - tau_G,
        with (nu_G, tau_G) from geometric_mean_lognormal_params
    """
    nu_G, tau_G = geometric_mean_lognormal_params(T, M, r, sigma, S_0)
    d_1 = (nu_G - np.log(strike) + tau_G**2) / tau_G
    d_2 = d_1 - tau_G
    return float(np.exp(nu_G + 0.5 * tau_G**2) * _std_normal_cdf(d_1)
                 - strike * _std_normal_cdf(d_2)) * np.exp(-r*T)


def _std_normal_cdf(x: float) -> float:
    """Standard normal CDF Phi(x) = (1 + erf(x / sqrt(2))) / 2."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def sample_discounted_geometric_asian_aux(N_prime: int, T: float, M: int, r: float, sigma: float,
                               S_0: float, strike: float,
                               rng: np.random.Generator) -> np.ndarray:
    """
    Parameters
    ----------
    N_prime : size N' of the auxiliary sample
    T : Maturity (payment date of the option)
    M : Number of discretization steps
    r : risk-free rate
    sigma : volatility term
    S_0 : initial price of the underlying
    strike : strike price K of the option
    rng : np.random.Generator that allows to fix the random values

    Returns
    -------
    a numpy array (N',) of auxiliary samples C'_j = exp(-r T) * max(G'_j - K, 0)
    of the discounted geometric Asian call payoff, where each G'_j is drawn
    from a single normal variable log G'_j ~ N(nu_G, tau_G^2), independently
    of the main sample
    """
    nu_G, tau_G = geometric_mean_lognormal_params(T, M, r, sigma, S_0)
    G = np.exp(nu_G + tau_G * rng.standard_normal(N_prime))
    return np.exp(-r*T)*np.maximum(G - strike, 0.0)

    
def estimate_control_mean(aux_samples: np.ndarray) -> float:
    """
    Parameters
    ----------
    aux_samples : numpy array (N',) of auxiliary samples C'_j, j = 1, ..., N'
        (e.g. from sample_discounted_geometric_asian_aux)

    Returns
    -------
    mu_C_hat : sample mean of the auxiliary samples C'_1, ..., C'_{N'},
        used as an estimate of mu_C in the CVEM estimator
    """
    return float(np.mean(aux_samples))

def compute_moments(A, C):
    cov_matrix = np.cov(A, C)
    var_A = cov_matrix[0, 0]                                 # Var(A)
    var_C = cov_matrix[1, 1]                                 # Var(C)
    cov_AC = cov_matrix[0, 1]                                # Cov(A, C)
    return var_A, var_C, cov_AC


def compute_optimal_b(covariance_AC: float, variance_C: float, N: int, N_prime: int) -> float:
    """
    Parameters
    ----------
    covariance_AC : covariance Cov(A, C) of the target A and the control C
        (true value or an estimate)
    variance_C : variance Var(C) of the control C (true value or an estimate)
    N : size of the main sample (A_i, C_i), i = 1, ..., N
    N_prime : size of the auxiliary sample C'_j, j = 1, ..., N', used to estimate mu_C

    Returns
    -------
    b : MSE-optimal control-variate coefficient when mu_C is replaced by an
        independent estimate from N' samples,
            b = Cov(A, C) / Var(C) * N' / (N + N'),
        i.e. the classical coefficient Cov(A, C) / Var(C) shrunk by the factor
        1 / (1 + N / N'). With estimated moments the result is an estimate b_hat.
    """
    if variance_C == 0: 
        warnings.warn("variance_C=0, division by zero, b=0 was returned (fallback to MC)")
        return 0.0
    return (covariance_AC/variance_C) * (N_prime / (N + N_prime))


def cvem_estimate(A: np.ndarray, C: np.ndarray, b: float, mu_C: float):
    """
    Parameters
    ----------
    A : numpy array (N,) of target samples A_i, i = 1, ..., N
    C : numpy array (N,) of control samples C_i computed on the same paths as A_i
    b : control-variate coefficient (e.g. from compute_optimal_b)
    mu_C : mean of the control, either the exact mu_C (classical CV)
        or its estimate mu_C_hat from the auxiliary sample (CVEM)

    Returns
    -------
    theta_hat : control-variate estimate of theta = E[A],
            theta_hat = mean(A) - b * (mean(C) - mu_C)
    """
    return float(np.mean(A) - b * (np.mean(C) - mu_C))


if __name__ == "__main__":
    # Tests
    N = 100
    M = 252
    T = 0.25                # maturity
    r = 0.05                # risk-free rate
    S_0 = 50                # initial price of the underlying
    sigma = 0.3             # volatility
    K = 50
    rng = np.random.default_rng(1)
    log_paths = sample_GBM_log_paths(N, T, M, r, sigma, S_0, rng)
    print(log_paths.shape)

    aux_samples = sample_discounted_geometric_asian_aux(10, T, M, r, sigma, S_0, K, rng)
    print(aux_samples)

    mu_C = estimate_control_mean(aux_samples)
    print(mu_C)
    

