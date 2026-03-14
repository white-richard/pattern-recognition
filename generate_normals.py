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
    mean: tuple[float, float] = (0, 0),
    std: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Generate 2D Gaussian data using independent standard deviations."""
    points = []

    for _ in range(num_samples):
        g1, g2 = gaussian_deviate()

        x_val = g1 * std[0] + mean[0]
        y_val = g2 * std[1] + mean[1]

        points.append([x_val, y_val])

    return np.array(points)


def plot_2d_gas(points: tuple[np.ndarray], fig_path: str = "gas.png") -> None:
    """Plot 2D Gaussian data."""
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    for i, point in enumerate(points):
        x = point[:, 0]
        y = point[:, 1]
        plt.scatter(x, y, s=1, alpha=0.1, label=f"N(μ_{i + 1}, Σ_{i + 1})", color=f"C{i}")
    plt.title("Scatter of generated 2D Gaussian.")
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
    points1 = generate_2d_gas_data(60000, mean=(1, 1), std=(1.0, 1.0))
    points2 = generate_2d_gas_data(140000, mean=(4, 4), std=(1.0, 1.0))
    path = plot_dir / "plotted_dataset_A.png"
    plot_2d_gas((points1, points2), fig_path=path)
