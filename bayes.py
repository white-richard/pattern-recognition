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


def bhattacharyya_distance_same_covariance(
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


def bhattacharyya_error_bound(
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
    k_05 = bhattacharyya_distance_same_covariance(mean1=mean1, mean2=mean2, covariance=covariance)
    v1 = np.sqrt(p1 * p2)
    v2 = np.exp(-k_05)
    error_bound = v1 * v2
    return float(error_bound)


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
