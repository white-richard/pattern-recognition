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
