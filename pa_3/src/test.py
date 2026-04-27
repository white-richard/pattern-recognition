import pathlib

import matplotlib.pyplot as plt
import numpy as np


def _normalize_img(img: np.ndarray) -> np.ndarray:
    if img.max() == img.min():
        return np.zeros_like(img)
    return (img - img.min()) / (img.max() - img.min())


def save_match_examples(
    *,
    train_ds: dict,
    test_ds: dict,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    correct_pairs: list[tuple[int, int]],
    incorrect_pairs: list[tuple[int, int]],
    output_dir: pathlib.Path,
    prefix: str,
) -> None:
    output_dir.mkdir(exist_ok=True)

    def _save_pairs(pairs: list[tuple[int, int]], filename: str) -> None:
        n_show = min(3, len(pairs))
        if n_show == 0:
            return
        plt.figure(figsize=(8, 4 * n_show))
        for row, (query_idx, match_idx) in enumerate(pairs[:n_show]):
            query_img = _normalize_img(test_ds["imgs"][query_idx])
            match_img = _normalize_img(train_ds["imgs"][match_idx])

            plt.subplot(n_show, 2, row * 2 + 1)
            plt.imshow(query_img, cmap="gray")
            plt.axis("off")

            plt.subplot(n_show, 2, row * 2 + 2)
            plt.imshow(match_img, cmap="gray")
            plt.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_dir / filename)
        plt.close()

    _save_pairs(correct_pairs, f"{prefix}_correct_matches.png")
    _save_pairs(incorrect_pairs, f"{prefix}_incorrect_matches.png")


def plot(datasets: dict, train: str, output_dir=pathlib.Path("attachments")) -> None:
    output_dir.mkdir(exist_ok=True)
    train_ds = datasets[train]

    # Save to fig the average face
    ef = train_ds["avg_face"]
    # Normalize to [0, 1]
    ef = (ef - ef.min()) / (ef.max() - ef.min())
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1)
    plt.imshow(ef, cmap="gray")
    # plt.title("Average Face")
    plt.axis("off")
    plt.savefig(output_dir / f"{train}_average_face.png")
    plt.close()

    # and The eigenfaces corresponding to the 10 largest eigenvalues
    num_components = train_ds["eigenvectors"].shape[1]
    n_show = min(10, num_components)
    plt.figure(figsize=(20, 20))
    for i in range(n_show):
        plt.subplot(1, n_show, i + 1)
        ef = train_ds["eigenvectors"][:, i].reshape(train_ds["avg_face"].shape)
        ef = (ef - ef.min()) / (ef.max() - ef.min())
        plt.imshow(
            ef,
            cmap="gray",
        )
        # plt.title(f"Eigenface {i + 1}")
        plt.axis("off")
    plt.savefig(output_dir / f"{train}_eigenfaces.png")
    # The eigenfaces corresponding to the 10 smallest eigenvalues
    plt.figure(figsize=(20, 20))
    for i in range(n_show):
        plt.subplot(1, n_show, i + 1)
        ef = train_ds["eigenvectors"][:, -i - 1].reshape(train_ds["avg_face"].shape)
        ef = (ef - ef.min()) / (ef.max() - ef.min())
        plt.imshow(
            ef,
            cmap="gray",
        )
        # plt.title(f"Eigenface {num_components - i}")
        plt.axis("off")
    plt.savefig(output_dir / f"{train}_eigenfaces_small.png")
    plt.close()


def main(*, train: str, test: str, pc_info: float, r: int = 5) -> None:

    datasets = np.load("attachments/datasets.npy", allow_pickle=True).item()
    train_ds = datasets[train]
    test_ds = datasets[test]
    print("Length of train and test datasets:", len(train_ds["labels"]), len(test_ds["labels"]))

    # Calculate number of components based on pc_info
    sum_eigen_values = np.sum(train_ds["eigenvalues"])
    ev_info = train_ds["eigenvalues"] / sum_eigen_values

    # Filter number of components
    idx = ev_info.cumsum() <= pc_info
    train_ds["num_components"] = np.sum(idx)
    print(
        f"Number of components to preserve {pc_info * 100:.2f}% info: {train_ds['num_components']}",
    )
    train_ds["eigenvectors"] = train_ds["eigenvectors"][:, idx]
    train_ds["eigenvalues"] = train_ds["eigenvalues"][idx]
    train_ds["projected"] = np.stack(train_ds["projections"], axis=0)[:, idx]

    # For each test image, project to train's eigenvector space
    test_ds["projected"] = np.dot(test_ds["mean_sub_imgs"], train_ds["eigenvectors"])

    labels = np.array(train_ds["labels"])
    test_labels = np.array(test_ds["labels"])
    correct_at_k = np.zeros(r, dtype=int)
    correct_pairs: list[tuple[int, int]] = []
    incorrect_pairs: list[tuple[int, int]] = []

    # Loop through each projected test img as the query
    for i, query in enumerate(test_ds["projected"]):
        # Mahalanobis distance
        distances = np.sum((train_ds["projected"] - query) ** 2 / train_ds["eigenvalues"], axis=1)
        top_idx = np.argsort(distances)[:r]
        true_label = test_labels[i]
        best_idx = top_idx[0]

        if labels[best_idx] == true_label:
            correct_pairs.append((i, best_idx))
        else:
            incorrect_pairs.append((i, best_idx))

        for k in range(r):
            if true_label in labels[top_idx[: k + 1]]:
                correct_at_k[k] += 1

    total = len(test_labels)
    correct = correct_at_k[0]
    incorrect = total - correct
    print(
        f"Correct: {correct}, Incorrect: {incorrect}, Acc: {correct / total:.2f}",
    )
    for k in range(r):
        acc = correct_at_k[k] / total
        print(f"Top-{k + 1} Acc: {acc:.2f} ({correct_at_k[k]}/{total})")

    # Plot CMC curve for ranks 1 to r
    ranks = np.arange(1, r + 1)
    cmc = correct_at_k / total
    plt.figure()
    plt.plot(ranks, cmc, marker="o")
    plt.xlabel("Rank")
    plt.ylabel("Accuracy")
    # plt.title(f"Comparative CMC Curve for {train} to {test}")
    plt.grid(True, linestyle="--")
    plt.ylim(cmc.min(), 1.0)
    plt.savefig(pathlib.Path("attachments") / f"{train}_to_{test}_cmc.png")
    plt.close()

    save_match_examples(
        train_ds=train_ds,
        test_ds=test_ds,
        train_labels=labels,
        test_labels=test_labels,
        correct_pairs=correct_pairs,
        incorrect_pairs=incorrect_pairs,
        output_dir=pathlib.Path("attachments"),
        prefix=f"{train}_to_{test}",
    )

    plot(datasets, train)


if __name__ == "__main__":
    print("=== Testing... ===")

    train = "fa_H"
    test = "fb_H"
    # pc_info = float(input("Enter the amount of information to be preserved (i.e. 0.8): "))
    pc_info = 0.8
    r = 50
    if not (0 < pc_info <= 1):
        print("Invalid input. Please enter a number between 0 and 1.")
    main(train=train, test=test, pc_info=pc_info, r=r)
