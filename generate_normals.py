import pathlib

import matplotlib.pyplot as plt
import numpy as np


def gaussian_deviate() -> (float, float):

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
    num_samples, mean=(0, 0), std=([1, 0], [0, 1])
) -> (list[float], list[float]):
    x = []
    y = []

    for _ in range(num_samples):
        g1, g2 = gaussian_deviate()

        x.append(g1 * std[0][0] + mean[0])
        y.append(g2 * std[1][1] + mean[1])

    return x, y


def plot_2d_gas(x1, y1, x2, y2, fig_name="gas.png") -> None:

    plt.figure()
    plt.scatter(x1, y1, alpha=0.3, color="blue", label="N(μ_1, Σ_1)")
    plt.scatter(x2, y2, alpha=0.3, color="orange", label="N(μ_1, Σ_2)")
    plt.title("Scatter of generated 2D Gaussian.")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.savefig("tmp.png")
    plt.close()


if __name__ == "__main__":
    x1, y1= generate_2d_gas_data(60000, mean=(1,1))
    x2, y2= generate_2d_gas_data(14000, mean=(4,4))

    path = pathlib.Path("tmp/gas.png")
    plot_2d_gas(x1, y1, x2, y2, fig_name=path)
