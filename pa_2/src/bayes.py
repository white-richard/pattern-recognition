import numpy as np


def sample_priors() -> tuple[np.ndarray, np.ndarray]:
    """Temp function to generate sample data for two priors."""
    rng = np.random.default_rng(seed=42)
    prior1_data = rng.choice([0, 1], size=500, p=[0.7, 0.3])
    prior2_data = rng.choice([0, 1], size=500, p=[0.2, 0.8])
    return prior1_data, prior2_data


def joint_prob_from_data(*, p1_data: np.ndarray, p2_data: np.ndarray) -> float:
    """Calculate the joint probability P(P1, P2) from data."""
    # Filter where both are true
    both = (p1_data == 1) & (p2_data == 1)
    return np.mean(both)


def cond_prob_from_data(*, target: np.ndarray, condition: np.ndarray) -> float:
    """Calculate the conditional probability P(Target | Condition) from data."""
    # Filter where condition is true in the target
    p1_g_p2 = target[condition == 1]
    return np.mean(p1_g_p2) if len(p1_g_p2) > 0 else 0


def bayes(*, p_b_g_p_a: float, p_a: float, p_b: float) -> float:
    """Calculate P(A | B) using Bayes' theorem."""
    return p_b_g_p_a * p_a / p_b


def discriminate_case_one(*, x: np.ndarray, p_i: float, mean: np.ndarray, std: float) -> float:
    """Calculate the discriminate for class c under the case one assumptions.

    Case 1 assumes the priors hold the same covariance matrix.
    We can then discriminate using:
        g_i(x) = ||x - µ_i||^2 / 2σ^2 + ln(P(ω_i))
    """
    # Axis 1 to compute across columns (features) for each sample
    t1 = -(np.linalg.norm(x - mean, axis=1) ** 2) / (2 * std**2)
    t2 = np.log(p_i)
    return t1 + t2


def discriminate_case_three(*, x: np.ndarray, p: float, mean: np.ndarray, cov: np.ndarray) -> float:
    """Calculate the discriminate for class c under the case three assumptions.

    Case 3 assumes arbritray covariance matrix.
    We can then discriminate using:
        g_i(x) = x^T@W_i@x+w_i^T@x+w_i0
        where:
            W_i = -∑_i^-1 / 2
            w_i = ∑_i^-1@µ_i
            w_i0 = -ln(|∑_i|) / 2 + ln(P(ω_i))
            w_i0 = -µ_i^T @ ∑_i^-1 @ µ_i / 2 - ln(|∑_i|) / 2 + ln(P(ω_i))
    """
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)

    W_i = -0.5 * inv_cov
    w_i = inv_cov @ mean
    w_i0 = -0.5 * (mean.T @ inv_cov @ mean) - 0.5 * np.log(det_cov) + np.log(p)

    quadratic_part = np.sum((x @ W_i) * x, axis=1)
    linear_part = x @ w_i

    return quadratic_part + linear_part + w_i0


def discriminate_euclidean(*, x: np.ndarray, mean: np.ndarray) -> float:
    """Calculate the discriminate for class c using euclidean distance.

    Formula:
        g_i(x) = - ||x - µ_i||^2
    """
    return -(np.linalg.norm(x - mean, axis=1) ** 2)


def bhattacharyya_distance_same_covariance_case_one(
    *,
    mean1: np.ndarray,
    mean2: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Calculate the case one simplified Bhattacharyya distance.

    Assumes each covariance matrix is equal, but not unit variance.
    If it was unit variance, we could switch inverse with transpose.
    The k(B) formula can be simplified because the second term on right
    becomes 0 since covariance is equal. The first term, two of the sigma's
    cancel out. The resulting formula looks like:
        k(0.5) = 1/8(µ_1 - µ_2)^T * ∑^-1 * (µ_1 - µ_2)
    """
    mean_diff = mean1 - mean2
    k_05 = (mean_diff.T @ np.linalg.inv(covariance) @ mean_diff) / 8
    return float(k_05)


def bhattacharyya_error_bound_case_one(
    *,
    mean1: np.ndarray,
    mean2: np.ndarray,
    covariance: np.ndarray,
    p1: float,
    p2: float,
) -> float:
    """Calculate the Bhattacharyya error bound for case one (equal covariance).

    Return the upper bound of the probability of error.
    We calculate with:
        P(error) <= sqrt(P(ω_1) * P(ω_2))e^(-k_0.5).
    """
    k_05 = bhattacharyya_distance_same_covariance_case_one(
        mean1=mean1,
        mean2=mean2,
        covariance=covariance,
    )
    v1 = np.sqrt(p1 * p2)
    v2 = np.exp(-k_05)
    error_bound = v1 * v2
    return float(error_bound)


def bhattacharyya_error_bound_case_three(  # noqa: PLR0913
    mean1: np.ndarray,
    mean2: np.ndarray,
    cov1: np.ndarray,
    cov2: np.ndarray,
    p1: float,
    p2: float,
) -> float:
    """Full Bhattacharyya bound for arbitrary covariances for case three."""
    diff = mean1 - mean2
    cov_avg = (cov1 + cov2) / 2

    k_mean = (0.125) * (diff.T @ np.linalg.inv(cov_avg) @ diff)

    det_1 = np.linalg.det(cov1)
    det_2 = np.linalg.det(cov2)
    det_avg = np.linalg.det(cov_avg)

    k_shape = 0.5 * np.log(det_avg / np.sqrt(det_1 * det_2))

    k_half = k_mean + k_shape

    # Final Error Bound
    return np.sqrt(p1 * p2) * np.exp(-k_half)


"""
Since we assume that the underlying distributions are guassians,
ML reduces to the sample mean and variance for the univariate case.
"""


def est_sample_mean(samples: np.ndarray) -> np.ndarray:
    """Calculate sample mean across all samples."""
    length = samples.shape[0]  # num of rows
    return np.sum(samples, axis=0) / length


def est_sample_cov(samples: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Calculate sample covariance across all samples."""
    length = samples.shape[0]  # num of rows
    difference = samples - mean  # n,d
    return difference.T @ difference / length  # d,d


def maximum_gaus_value_c(cov: np.ndarray) -> float:
    """Calculate the maximum value of the Gaussian distribution for a given covariance."""
    det_cov = np.linalg.det(cov)
    return 1.0 / (2 * np.pi * np.sqrt(det_cov))


def mult_gaussian_discriminant(x, mean, cov):
    """Calculate the multivariate Gaussian discriminant for given samples, mean, and covariance.

    Formula from textbook.
    """
    d = 2

    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)

    # c = 1 / (2 * pi * |cov|^1/2)
    norm = 1.0 / ((2 * np.pi) ** (d / 2.0) * np.sqrt(det_cov))

    # x - mean
    x_minus_mean = x - mean

    # -0.5 * (x-mean)^T * inv_cov * (x-mean)
    exponent = -0.5 * np.sum(x_minus_mean @ inv_cov * x_minus_mean, axis=1)

    return norm * np.exp(exponent)


if __name__ == "__main__":
    # Known
    p1_data, p2_data = sample_priors()
    p1 = np.mean(p1_data)
    p2 = np.mean(p2_data)
    p2_g_p1 = cond_prob_from_data(target=p2_data, condition=p1_data)

    p1_g_p2 = bayes(p_bGp_a=p2_g_p1, p_a=p1, p_b=p2)
    print(f"Prior 1: {p1}")
    print(f"Prior 2: {p2}")
    print(f"P2 | P1: {p2_g_p1}")
    print(f"P1 | P2 (Conditional Posterior): {p1_g_p2}")
