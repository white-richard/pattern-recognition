"""Pretty dimensionality reduction demo: t-SNE on a Swiss roll.

This script shows how t-SNE (a non-linear method) better preserves
local neighborhoods on a Swiss roll compared to linear projections.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import TSNE
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


def _safe_perplexity(perplexity: float, n_samples: int) -> float:
    """Clamp perplexity to a safe range for the given sample size."""
    if n_samples <= 2:
        return 1.0
    max_reasonable = max(5.0, n_samples / 3)
    return min(perplexity, n_samples - 1, max_reasonable)


def _find_local_cluster(x_proj: np.ndarray, t: np.ndarray, k: int = 25) -> np.ndarray | None:
    """Find a compact local neighborhood with similar intrinsic parameter values."""
    if x_proj.shape[0] < 3:
        return None

    k = max(5, min(k, x_proj.shape[0]))
    nbrs = NearestNeighbors(n_neighbors=k).fit(x_proj)
    _distances, indices = nbrs.kneighbors(x_proj)

    t_range = float(np.ptp(t))
    if t_range == 0:
        return None

    spreads = np.array([np.ptp(t[idx]) / t_range for idx in indices])
    best_idx = int(np.argmin(spreads))
    return indices[best_idx]


def plot_tsne_on_swiss_roll(
    n_samples: int = 3000,
    noise: float = 0.2,
    seed: int = 7,
    perplexity: float = 35.0,
    max_iter: int = 1000,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Create a side-by-side plot of Swiss roll and its t-SNE projection."""
    _set_plot_style()

    x, t = _make_swiss_roll(n_samples=n_samples, noise=noise, seed=seed)

    perplexity = _safe_perplexity(perplexity, n_samples)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=max_iter,
        random_state=seed,
    )
    x_tsne = tsne.fit_transform(x)

    fig = plt.figure(figsize=(12.5, 5.6), dpi=160)
    fig.suptitle("Non-Linear Structure Is Unrolled", y=1.02)

    ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_2d = fig.add_subplot(1, 2, 2)

    norm = mpl.colors.Normalize(vmin=float(t.min()), vmax=float(t.max()))
    scatter_kwargs = {
        "c": t,
        "cmap": "viridis",
        "norm": norm,
        "s": 10,
        "alpha": 0.9,
        "linewidths": 0,
    }

    ax_3d.scatter(x[:, 0], x[:, 1], x[:, 2], **scatter_kwargs)
    ax_3d.set_title("Original Swiss Roll (3D)")
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    ax_3d.view_init(elev=20, azim=120)
    ax_3d.grid(False)

    ax_2d.scatter(x_tsne[:, 0], x_tsne[:, 1], **scatter_kwargs)
    ax_2d.set_title("t-SNE Embedding (2D)")
    ax_2d.set_xlabel("Dim 1")
    ax_2d.set_ylabel("Dim 2")

    cluster = _find_local_cluster(x_tsne, t, k=25)
    if cluster is not None:
        pts = x_tsne[cluster]
        center = pts.mean(axis=0)
        radius = np.max(np.linalg.norm(pts - center, axis=1)) * 1.15
        ax_2d.add_patch(
            Circle(
                center,
                radius,
                edgecolor="#d1495b",
                facecolor="none",
                linewidth=1.8,
                alpha=0.9,
                zorder=5,
            ),
        )
        ax_2d.annotate(
            "Local neighborhoods\npreserved",
            xy=(center[0], center[1]),
            xytext=(24, 24),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "#333333"},
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#333333", "alpha": 0.9},
        )

    fig.subplots_adjust(right=0.88, wspace=0.25)
    cax = fig.add_axes([0.9, 0.17, 0.02, 0.66])
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="viridis"),
        cax=cax,
    )
    cbar.set_label("Intrinsic roll parameter (t)")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="t-SNE on Swiss roll visualization")
    parser.add_argument("--samples", type=int, default=3000, help="Number of samples")
    parser.add_argument("--noise", type=float, default=0.2, help="Noise level")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--perplexity", type=float, default=35.0, help="t-SNE perplexity")
    parser.add_argument("--iterations", type=int, default=1000, help="t-SNE iterations")
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("outputs/tsne_swiss_roll.png"),
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
    plot_tsne_on_swiss_roll(
        n_samples=args.samples,
        noise=args.noise,
        seed=args.seed,
        perplexity=args.perplexity,
        max_iter=args.iterations,
        save_path=args.save,
        show=not args.no_show,
    )
