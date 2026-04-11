import pathlib

import numpy as np
from bayes import (
    bhattacharyya_error_bound_case_one,
    bhattacharyya_error_bound_case_three,
    discriminate_case_one,
    discriminate_case_three,
    est_sample_cov,
    est_sample_mean,
)
from generate_normals import generate_2d_gas_data, plot_2d_gas
from seed import set_all_seeds


def discriminate(
    points,
    distribution,
    labels,
    total,
    points_by_class,
    fig_dir,
    case: int = 3,
) -> None:

    if case == 3:
        g1 = discriminate_case_three(
            x=points,
            p=distribution[1]["p"],
            mean=distribution[1]["mean"],
            cov=distribution[1]["cov"],
        )
        g2 = discriminate_case_three(
            x=points,
            p=distribution[2]["p"],
            mean=distribution[2]["mean"],
            cov=distribution[2]["cov"],
        )
    elif case == 1:
        g1 = discriminate_case_one(
            x=points,
            p_i=distribution[1]["p"],
            mean=distribution[1]["mean"],
            std=np.sqrt(distribution[1]["cov"][0]),
        )
        g2 = discriminate_case_one(
            x=points,
            p_i=distribution[2]["p"],
            mean=distribution[2]["mean"],
            std=np.sqrt(distribution[2]["cov"][0]),
        )
    else:
        msg = f"Unsupported case: {case}"
        raise ValueError(msg)
    predictions = np.where(g1 > g2, 1, 2)

    # Class 1 error rate
    class1_end = distribution[1]["n"]
    num_errors_class1 = np.sum(predictions[:class1_end] != labels[:class1_end])
    error_rate_class1 = num_errors_class1 / distribution[1]["n"]
    print(f"Class 1 total errors: {num_errors_class1}")
    print(f"Class 1 error rate: {error_rate_class1 * 100:.2f}%")

    # Class 2 error rate
    num_errors_class2 = np.sum(predictions[class1_end:] != labels[class1_end:])
    error_rate_class2 = num_errors_class2 / distribution[2]["n"]
    print(f"Class 2 total errors: {num_errors_class2}")
    print(f"Class 2 error rate: {error_rate_class2 * 100:.2f}%")

    # Overall Error rate
    num_errors = np.sum(predictions != labels)
    error_rate = num_errors / total
    print(f"Total errors: {num_errors}")
    print(f"Minimum error rate: {error_rate * 100:.2f}%")

    def boundary_func(grid_pts) -> float:
        """Boundary function for two discriminate.

        From g_1(x) = g_2(x)
        """
        if case == 3:
            g1_grid = discriminate_case_three(
                x=grid_pts,
                p=distribution[1]["p"],
                mean=distribution[1]["mean"],
                cov=distribution[1]["cov"],
            )
            g2_grid = discriminate_case_three(
                x=grid_pts,
                p=distribution[2]["p"],
                mean=distribution[2]["mean"],
                cov=distribution[2]["cov"],
            )
        else:
            g1_grid = discriminate_case_one(
                x=grid_pts,
                p_i=distribution[1]["p"],
                mean=distribution[1]["mean"],
                std=np.sqrt(distribution[1]["cov"][0]),
            )
            g2_grid = discriminate_case_one(
                x=grid_pts,
                p_i=distribution[2]["p"],
                mean=distribution[2]["mean"],
                std=np.sqrt(distribution[2]["cov"][0]),
            )
        return g1_grid - g2_grid

    # Plot w/ decision boundary
    plot_2d_gas(
        points=(points_by_class[1], points_by_class[2]),
        decision_fn=boundary_func,
        fig_path=fig_dir / "experiment_1_decision.png",
    )

    # Calculate upper bound
    if case == 3:
        b_error = bhattacharyya_error_bound_case_three(
            mean1=distribution[1]["mean"],
            mean2=distribution[2]["mean"],
            cov1=distribution[1]["cov"],
            cov2=distribution[2]["cov"],
            p1=distribution[1]["p"],
            p2=distribution[2]["p"],
        )
    else:
        b_error = bhattacharyya_error_bound_case_one(
            mean1=distribution[1]["mean"],
            mean2=distribution[2]["mean"],
            covariance=np.diag(distribution[1]["cov"]),
            p1=distribution[1]["p"],
            p2=distribution[2]["p"],
        )
    print(f"Error upper bound: {b_error * 100:.2f}%")


def main(case: int = 3) -> None:
    rng1 = set_all_seeds(42)
    rng2 = np.random.default_rng(43)
    fig_dir = pathlib.Path("attachments")
    class_one_n = 60000
    class_two_n = 140000

    print("\n" + "=" * 5 + f"Experiment 1, Case {case}" + "=" * 5)

    # True class-conditional distribution
    true_distribution = {
        1: {
            "n": class_one_n,
            "mean": np.array([1, 1]),
            "cov": np.array([[1.0, 0.0], [0.0, 1.0]]),
        },
        2: {
            "n": class_two_n,
            "mean": np.array([4, 4]),
            "cov": np.array([[1.0, 0.0], [0.0, 1.0]]),
        },
    }
    class_ids = (1, 2)

    total = sum(true_distribution[c]["n"] for c in class_ids)

    # Add priors
    for c in class_ids:
        true_distribution[c]["p"] = true_distribution[c]["n"] / total

    true_labels = {c: np.full(true_distribution[c]["n"], c) for c in class_ids}

    # Generate points
    true_points_by_class = {
        c: generate_2d_gas_data(
            true_distribution[c]["n"],
            mean=true_distribution[c]["mean"],
            covariance=true_distribution[c]["cov"],
            rng=rng1 if c == 1 else rng2,  # Hacky
        )
        for c in class_ids
    }

    # Stack
    true_points = np.vstack([true_points_by_class[c] for c in class_ids])
    true_labels_all = np.concatenate([true_labels[c] for c in class_ids])

    # Est class 1
    est_mean_1 = est_sample_mean(true_points_by_class[1])
    est_cov_1 = est_sample_cov(true_points_by_class[1], est_mean_1)

    # Est class 2
    est_mean_2 = est_sample_mean(true_points_by_class[2])
    est_cov_2 = est_sample_cov(true_points_by_class[2], est_mean_2)

    # Est class-conditional distribution
    distribution = {
        1: {
            "n": class_one_n,
            "mean": est_mean_1,
            "cov": est_cov_1,
        },
        2: {
            "n": class_two_n,
            "mean": est_mean_2,
            "cov": est_cov_2,
        },
    }

    # Case 1 and 2 assume zeros on the diagonal for our problem
    # we can just remove the off diagonal elements
    if case != 3:
        for c in class_ids:
            true_distribution[c]["cov"] = np.diag(true_distribution[c]["cov"])
            distribution[c]["cov"] = np.diag(distribution[c]["cov"])

    # Generate est points
    points_by_class = {
        c: generate_2d_gas_data(
            distribution[c]["n"],  # Still use the same count
            mean=distribution[c]["mean"],
            covariance=distribution[c]["cov"],
            rng=rng1 if c == 1 else rng2,  # Hacky
        )
        for c in class_ids
    }

    # Add priors
    for c in class_ids:
        distribution[c]["p"] = distribution[c]["n"] / total

    labels = {c: np.full(distribution[c]["n"], c) for c in class_ids}

    # Stack
    np.vstack([points_by_class[c] for c in class_ids])
    labels = np.concatenate([labels[c] for c in class_ids])

    # Discriminate
    print("==From true distribution:")
    discriminate(
        true_points,
        true_distribution,
        true_labels_all,
        total,
        true_points_by_class,
        fig_dir,
        case=case,
    )
    print("==From estimated distribution:")
    discriminate(
        true_points,
        distribution,
        true_labels_all,
        total,
        points_by_class,
        fig_dir,
        case=case,
    )


if __name__ == "__main__":
    cases = (1, 3)
    for case in cases:
        main(case)
