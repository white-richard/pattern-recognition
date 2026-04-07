import pathlib

import numpy as np
from bayes import bhattacharyya_error_bound_case_three, discriminate_case_three
from generate_normals import generate_2d_gas_data, plot_2d_gas
from seed import set_all_seeds

if __name__ == "__main__":
    set_all_seeds(42)
    fig_dir = pathlib.Path("attachments")

    print("\n" + "=" * 5 + "Experiment 2" + "=" * 5)

    # Class 1
    num_class1 = 60000
    mean_class1 = np.array([1, 1])
    cov1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels1 = np.full(num_class1, 1)

    # Class 2
    num_class2 = 140000
    mean_class2 = np.array([4, 4])
    cov2 = np.array([[4.0, 0.0], [0.0, 8.0]])
    labels2 = np.full(num_class2, 2)

    total = num_class1 + num_class2

    # Calculate priors
    p1 = num_class1 / total
    p2 = num_class2 / total

    # Generate out normal points
    points1 = generate_2d_gas_data(num_class1, mean=mean_class1, covariance=cov1)
    points2 = generate_2d_gas_data(num_class2, mean=mean_class2, covariance=cov2)

    # Stack our points and labels for distrimination
    points = np.vstack((points1, points2))
    labels = np.concatenate((labels1, labels2))

    # Discriminate
    g1 = discriminate_case_three(x=points, p=p1, mean=mean_class1, cov=cov1)
    g2 = discriminate_case_three(x=points, p=p2, mean=mean_class2, cov=cov2)
    predictions = np.where(g1 > g2, 1, 2)

    # Class 1 error rate
    num_errors_class1 = np.sum(predictions[:num_class1] != labels1)
    error_rate_class1 = num_errors_class1 / num_class1
    print(f"Class 1 total errors: {num_errors_class1}")
    print(f"Class 1 error rate: {error_rate_class1 * 100:.2f}%")

    # Class 2 error rate
    num_errors_class2 = np.sum(predictions[num_class1:] != labels2)
    error_rate_class2 = num_errors_class2 / num_class2
    print(f"Class 2 total errors: {num_errors_class2}")
    print(f"Class 2 error rate: {error_rate_class2 * 100:.2f}%")

    # Overall Error rate
    num_errors = np.sum(predictions != labels)
    error_rate = num_errors / total
    print(f"Total errors: {num_errors}")
    print(f"Minimum error rate: {error_rate * 100:.2f}%")

    def boundary_func(grid_pts) -> float:  # noqa: ANN001
        """Boundary function for two discriminate.

        From g_1(x) = g_2(x)
        """
        g1_grid = discriminate_case_three(x=grid_pts, p=p1, mean=mean_class1, cov=cov1)
        g2_grid = discriminate_case_three(x=grid_pts, p=p2, mean=mean_class2, cov=cov2)
        return g1_grid - g2_grid

    # Plot w/ decision boundary
    plot_2d_gas(
        points=(points1, points2),
        decision_fn=boundary_func,
        fig_path=fig_dir / "experiment2_decision.png",
    )

    # Calculate upper bound
    b_error = bhattacharyya_error_bound_case_three(
        mean1=np.array(mean_class1),
        mean2=np.array(mean_class2),
        cov1=cov1,
        cov2=cov2,
        p1=p1,
        p2=p2,
    )
    print(f"Error upper bound: {b_error * 100:.2f}%")
