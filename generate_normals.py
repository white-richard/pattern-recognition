import matplotlib.pyplot as plt
import numpy as np
import pathlib


def gaussian_deviate(dim: int = 2) -> (float, float):

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

    if dim == 1:
        return gas1
    return gas1, gas2


def sample_gas_dev(num_gas: int, num_gas2: int = 0) -> (list[float], list[float]):
    g1_list = []
    g2_list = []

    # Sample from each
    for idx in range(max(num_gas, num_gas2)):
        g1, g2 = gaussian_deviate(dim=2)

        if idx <= num_gas - 1:
            g1_list.append(g1)

        if idx <= num_gas2 - 1:
            g2_list.append(g2)

    return g1_list, g2_list


def plot_gasdev(g1_list, g2_list, fig_name="gas.png") -> None:

    if len(g1_list) != len(g2_list):
        raise ValueError("Lists must have the same length.")

    plt.figure()
    plt.scatter(g1_list, g2_list, alpha=0.3)
    plt.title("Scatter of generated 2D Gaussian.")
    plt.xlabel("g1")
    plt.ylabel("g2")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig("tmp.png")


if __name__ == "__main__":
    sample_gas_dev(num_gas=60000, num_gas2=140000)

    g1_list, g2_list = sample_gas_dev(num_gas=60000, num_gas2=140000)
    
    path = pathlib.Path("tmp/gas.png")
    plot_gasdev(g1_list, g2_list, fig_name=path)
