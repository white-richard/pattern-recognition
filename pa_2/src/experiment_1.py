import pathlib

import numpy as np
from bayes import bhattacharyya_error_bound_case_three, discriminate_case_three
from generate_normals import generate_2d_gas_data, plot_2d_gas
from seed import set_all_seeds

if __name__ == "__main__":
    rng1 = set_all_seeds(42)
    rng2 = np.random.default_rng(43)
    fig_dir = pathlib.Path("attachments")

    print("\n" + "=" * 5 + "Experiment 1" + "=" * 5)

    # True class-conditional distribution
    true_distribution = {
        1: {
            "n": 60000,
            "mean": np.array([1, 1]),
            "cov": np.array([[1.0, 0.0], [0.0, 1.0]]),
        },
        2: {
            "n": 140000,
            "mean": np.array([4, 4]),
            "cov": np.array([[4.0, 0.0], [0.0, 8.0]]),
        },
    }
    class_ids = (1, 2)

    total = sum(true_distribution[c]["n"] for c in class_ids)

    # Add priors
    for c in class_ids:
        true_distribution[c]["p"] = true_distribution[c]["n"] / total

    labels = {c: np.full(true_distribution[c]["n"], c) for c in class_ids}

    # Generate points
    points_by_class = {
        c: generate_2d_gas_data(
            true_distribution[c]["n"],
            mean=true_distribution[c]["mean"],
            covariance=true_distribution[c]["cov"],
            rng=rng1 if c == 1 else rng2,  # Hacky
        )
        for c in class_ids
    }

    # Stack
    points = np.vstack([points_by_class[c] for c in class_ids])
    labels_all = np.concatenate([labels[c] for c in class_ids])

    # Discriminate
    g1 = discriminate_case_three(
        x=points,
        p=true_distribution[1]["p"],
        mean=true_distribution[1]["mean"],
        cov=true_distribution[1]["cov"],
    )
    g2 = discriminate_case_three(
        x=points,
        p=true_distribution[2]["p"],
        mean=true_distribution[2]["mean"],
        cov=true_distribution[2]["cov"],
    )
    predictions = np.where(g1 > g2, 1, 2)

    # Class 1 error rate
    class1_end = true_distribution[1]["n"]
    num_errors_class1 = np.sum(predictions[:class1_end] != labels[1])
    error_rate_class1 = num_errors_class1 / true_distribution[1]["n"]
    print(f"Class 1 total errors: {num_errors_class1}")
    print(f"Class 1 error rate: {error_rate_class1 * 100:.2f}%")

    # Class 2 error rate
    num_errors_class2 = np.sum(predictions[class1_end:] != labels[2])
    error_rate_class2 = num_errors_class2 / true_distribution[2]["n"]
    print(f"Class 2 total errors: {num_errors_class2}")
    print(f"Class 2 error rate: {error_rate_class2 * 100:.2f}%")

    # Overall Error rate
    num_errors = np.sum(predictions != labels_all)
    error_rate = num_errors / total
    print(f"Total errors: {num_errors}")
    print(f"Minimum error rate: {error_rate * 100:.2f}%")

    def boundary_func(grid_pts) -> float:
        """Boundary function for two discriminate.

        From g_1(x) = g_2(x)
        """
        g1_grid = discriminate_case_three(
            x=grid_pts,
            p=true_distribution[1]["p"],
            mean=true_distribution[1]["mean"],
            cov=true_distribution[1]["cov"],
        )
        g2_grid = discriminate_case_three(
            x=grid_pts,
            p=true_distribution[2]["p"],
            mean=true_distribution[2]["mean"],
            cov=true_distribution[2]["cov"],
        )
        return g1_grid - g2_grid

    # Plot w/ decision boundary
    plot_2d_gas(
        points=(points_by_class[1], points_by_class[2]),
        decision_fn=boundary_func,
        fig_path=fig_dir / "experiment2_decision.png",
    )

    # Calculate upper bound
    b_error = bhattacharyya_error_bound_case_three(
        mean1=true_distribution[1]["mean"],
        mean2=true_distribution[2]["mean"],
        cov1=true_distribution[1]["cov"],
        cov2=true_distribution[2]["cov"],
        p1=true_distribution[1]["p"],
        p2=true_distribution[2]["p"],
    )
    print(f"Error upper bound: {b_error * 100:.2f}%")
