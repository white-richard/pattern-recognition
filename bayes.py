import numpy as np


def sample_priors():
    prior1_data = np.random.choice([0, 1], size=500, p=[0.7, 0.3])
    prior2_data = np.random.choice([0, 1], size=500, p=[0.2, 0.8])
    return prior1_data, prior2_data


def joint_prob(*, p1, p2):
    # Filter where both are true
    both = (p1 == 1) & (p2 == 1)
    return np.mean(both)


def cond_prob(*, target, condition):
    # Filter where condition is true in the target
    p1Gp2 = target[condition == 1]
    return np.mean(p1Gp2) if len(p1Gp2) > 0 else 0


def bayes(*, p_bGp_a, p_a, p_b):
    return p_bGp_a * p_a / p_b


if __name__ == "__main__":
    # Known
    p1_data, p2_data = sample_priors()
    p1 = np.mean(p1_data)
    p2 = np.mean(p2_data)
    p2Gp1 = cond_prob(target=p2_data, condition=p1_data)

    p1Gp2 = bayes(p_bGp_a=p2Gp1, p_a=p1, p_b=p2)
    print(f"Prior 1: {p1}")
    print(f"Prior 2: {p2}")
    print(f"P2 | P1: {p2Gp1}")
    print(f"P1 | P2 (Conditional Posterior): {p1Gp2}")
