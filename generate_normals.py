import pathlib

import matplotlib.pyplot as plt
import numpy as np


def gaussian_deviate() -> (float, float):
    """Generate two independent normally distributed random vars."""
    rng = np.random.default_rng()

    fac = r = v1 = v2 = 0.0

    while True:
        # Uniform sampler
        v1 = 2.0 * rng.random() - 1.0
        v2 = 2.0 * rng.random() - 1.0
        r = v1 * v1 + v2 * v2

        if r <= 1.0:
            break
    fac = np.sqrt(-2.0 * np.log(r) / r)
    gas1 = v2 * fac
    gas2 = v1 * fac

    return gas1, gas2


def generate_2d_gas_data(
    num_samples: int,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """Generate 2D Gaussian data using a covariance matrix and gaussian_deviate."""
    points = np.zeros((num_samples, 2))

    # Uhh we need to decompose the covariance matrix to fit in this
    chol = np.linalg.cholesky(covariance)

    for idx in range(num_samples):
        g1, g2 = gaussian_deviate()
        z = np.array([g1, g2])

        # Transform: x = Lz + mu
        points[idx] = chol @ z + mean

    return np.array(points)


def plot_2d_gas(
    points: tuple[np.ndarray],
    title: str = "Plotted 2D Gaussians",
    fig_path: str = "gas.png",
    decision_fn: callable[[np.ndarray], np.ndarray] | None = None,
) -> None:
    """Plot 2D Gaussian data."""
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    for i, point in enumerate(points):
        x = point[:, 0]
        y = point[:, 1]
        plt.scatter(x, y, s=1, alpha=0.1, label=f"N(μ_{i + 1}, Σ_{i + 1})", color=f"C{i}")

    if decision_fn is not None:
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()

        # Build grid for decision
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Calcluate boundary
        z = decision_fn(grid_points).reshape(xx.shape)

        plt.contour(xx, yy, z, levels=[0], colors="black", linestyles="--")
        plt.plot([], [], "k--", label="Decision Boundary")

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(visible=True)
    plt.legend()
    plt.savefig(fig_path)
    plt.close()


if __name__ == "__main__":
    plot_dir = pathlib.Path("attachments")

    # Generate dataset A
    points1 = generate_2d_gas_data(
        60000,
        mean=np.array([1, 1]),
        covariance=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    points2 = generate_2d_gas_data(
        140000,
        mean=np.array([4, 4]),
        covariance=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )

    path = plot_dir / "plotted_dataset_A.png"
    plot_2d_gas((points1, points2), fig_path=path)

    # Generate dataset B
    points1 = generate_2d_gas_data(
        60000,
        mean=np.array([1, 1]),
        covariance=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    points2 = generate_2d_gas_data(
        140000,
        mean=np.array([4, 4]),
        covariance=np.array([[4.0, 0.0], [0.0, 8.0]]),
    )
    path = plot_dir / "plotted_dataset_B.png"
    plot_2d_gas((points1, points2), fig_path=path)
