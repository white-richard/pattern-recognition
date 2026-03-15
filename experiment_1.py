import pathlib

import numpy as np

from bayes import bhattacharyya_error_bound, discriminate_case_one
from generate_normals import generate_2d_gas_data, plot_2d_gas

if __name__ == "__main__":
    fig_dir = pathlib.Path("attachments")

    # Class 1
    num_class1 = 60000
    mean_class1 = (1, 1)
    labels1 = np.full(num_class1, 1)

    # Class 2
    num_class2 = 140000
    mean_class2 = (4, 4)
    labels2 = np.full(num_class2, 2)

    std = 1
    cov_matrix = np.array([[1, 0], [0, 1]])
    total = num_class1 + num_class2

    # Calculate priors
    p1 = num_class1 / total
    p2 = num_class2 / total

    # Generate out normal points
    points1 = generate_2d_gas_data(num_class1, mean=mean_class1, covariance=cov_matrix)
    points2 = generate_2d_gas_data(num_class2, mean=mean_class2, covariance=cov_matrix)

    # Stack our points and labels for distrimination
    points = np.vstack((points1, points2))
    labels = np.concatenate((labels1, labels2))

    # Discriminate
    g1 = discriminate_case_one(x=points, p_i=p1, mean=mean_class1, std=std)
    g2 = discriminate_case_one(x=points, p_i=p2, mean=mean_class2, std=std)
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
        g1_grid = discriminate_case_one(x=grid_pts, p_i=p1, mean=mean_class1, std=std)
        g2_grid = discriminate_case_one(x=grid_pts, p_i=p2, mean=mean_class2, std=std)
        return g1_grid - g2_grid

    # Plot w/ decision boundary
    plot_2d_gas(
        points=(points1, points2),
        decision_fn=boundary_func,
        fig_path=fig_dir / "experiment1_decision.png",
    )

    # Calculate upper bound
    b_error = bhattacharyya_error_bound(
        mean1=np.array(mean_class1), mean2=np.array(mean_class2), covariance=cov_matrix, p1=p1, p2=p2
    )
    print(f"Error upper bound: {b_error * 100:.2f}%")
