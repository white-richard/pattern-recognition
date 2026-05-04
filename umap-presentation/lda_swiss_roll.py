"""Pretty dimensionality reduction demo: LDA on a Swiss roll.

This script uses binned Swiss roll coordinates as proxy classes to show
that LDA (a linear, supervised method) cannot untangle a non-linear
manifold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors


def _set_plot_style() -> None:
    """Apply a clean, slide-friendly plotting style."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "text.color": "#111111",
            "font.size": 12,
            "axes.titleweight": "semibold",
        },
    )


def _make_swiss_roll(n_samples: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate Swiss roll data and the intrinsic roll parameter t."""
    x, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)
    return x, t


def _make_class_labels(t: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Bin the intrinsic parameter into discrete classes (quantile bins)."""
    n_classes = max(2, int(n_classes))
    quantiles = np.linspace(0.0, 1.0, n_classes + 1)
    bins = np.quantile(t, quantiles)
    bins[0] -= 1e-6
    bins[-1] += 1e-6
    y = np.digitize(t, bins[1:-1])
    return y, bins


def _find_mixed_pairs(
    x_proj: np.ndarray,
    t: np.ndarray,
    n_pairs: int = 3,
    min_gap: float = 0.6,
    neighbor_k: int = 6,
) -> list[tuple[int, int]]:
    """Find point pairs that are close in LDA but far along the roll."""
    if x_proj.shape[0] < 3:
        return []

    k = min(neighbor_k, x_proj.shape[0] - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(x_proj)
    distances, indices = nbrs.kneighbors(x_proj)

    t_range = float(np.ptp(t))
    if t_range == 0:
        return []

    close_dist = np.percentile(distances[:, 1], 35)
    candidates: list[tuple[float, float, int, int]] = []
    for i in range(x_proj.shape[0]):
        for pos in range(1, indices.shape[1]):
            j = indices[i, pos]
            d = distances[i, pos]
            if d > close_dist:
                continue
            t_gap = abs(t[i] - t[j]) / t_range
            if t_gap < min_gap:
                continue
            candidates.append((t_gap, d, i, j))

    candidates.sort(key=lambda item: (-item[0], item[1]))

    pairs: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for _, _, i, j in candidates:
        key = tuple(sorted((i, j)))
        if key in used:
            continue
        used.add(key)
        pairs.append((i, j))
        if len(pairs) >= n_pairs:
            break

    return pairs


def plot_lda_on_swiss_roll(
    n_samples: int = 3000,
    noise: float = 0.2,
    seed: int = 7,
    n_classes: int = 6,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Create a side-by-side plot of Swiss roll and its LDA projection."""
    _set_plot_style()

    x, t = _make_swiss_roll(n_samples=n_samples, noise=noise, seed=seed)
    y, _bins = _make_class_labels(t, n_classes=n_classes)

    lda = LinearDiscriminantAnalysis(n_components=2)
    x_lda = lda.fit_transform(x, y)

    n_classes = int(y.max() + 1)
    cmap = mpl.cm.get_cmap("viridis", n_classes)
    bounds = np.arange(-0.5, n_classes + 0.5, 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(12.5, 5.6), dpi=160)
    fig.suptitle("Linear Separation Mixes the Roll", y=1.02)

    ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_2d = fig.add_subplot(1, 2, 2)

    scatter_kwargs = {
        "c": y,
        "cmap": cmap,
        "norm": norm,
        "s": 10,
        "alpha": 0.9,
        "linewidths": 0,
    }

    ax_3d.scatter(x[:, 0], x[:, 1], x[:, 2], **scatter_kwargs)
    ax_3d.set_title("Swiss Roll with Binned Classes (3D)")
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    ax_3d.view_init(elev=20, azim=120)
    ax_3d.grid(False)

    ax_2d.scatter(x_lda[:, 0], x_lda[:, 1], **scatter_kwargs)
    ax_2d.set_title("LDA Projection (2D)")
    ax_2d.set_xlabel("LD 1")
    ax_2d.set_ylabel("LD 2")

    mixed_pairs = _find_mixed_pairs(x_lda, t, n_pairs=3, min_gap=0.6)
    for i, j in mixed_pairs:
        ax_2d.plot(
            [x_lda[i, 0], x_lda[j, 0]],
            [x_lda[i, 1], x_lda[j, 1]],
            color="#d1495b",
            linewidth=1.4,
            alpha=0.9,
            zorder=4,
        )
        ax_2d.scatter(
            [x_lda[i, 0], x_lda[j, 0]],
            [x_lda[i, 1], x_lda[j, 1]],
            s=42,
            c=[y[i], y[j]],
            cmap=cmap,
            norm=norm,
            edgecolor="#111111",
            linewidth=0.6,
            zorder=5,
        )

    if mixed_pairs:
        i, j = mixed_pairs[0]
        mid = (x_lda[i] + x_lda[j]) / 2
        ax_2d.annotate(
            "Close in LDA,\nfar along roll",
            xy=(mid[0], mid[1]),
            xytext=(22, 24),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "#333333"},
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#333333", "alpha": 0.9},
        )

    fig.subplots_adjust(right=0.88, wspace=0.25)
    cax = fig.add_axes([0.9, 0.17, 0.02, 0.66])
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        ticks=np.arange(n_classes),
    )
    tick_labels = [f"Class {i + 1}" for i in range(n_classes)]
    cbar.set_ticklabels(tick_labels)
    cbar.set_label("Binned roll parameter (proxy classes)")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LDA on Swiss roll visualization")
    parser.add_argument("--samples", type=int, default=3000, help="Number of samples")
    parser.add_argument("--noise", type=float, default=0.2, help="Noise level")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--classes", type=int, default=6, help="Number of binned classes")
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("outputs/lda_swiss_roll.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a window; only save the figure",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_lda_on_swiss_roll(
        n_samples=args.samples,
        noise=args.noise,
        seed=args.seed,
        n_classes=args.classes,
        save_path=args.save,
        show=not args.no_show,
    )
