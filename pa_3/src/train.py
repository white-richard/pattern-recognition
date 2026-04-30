import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_img(path: pathlib.Path) -> np.ndarray | None:
    """Load ppm image."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float64) / 255.0  # Normalize to [0, 1]


def compute_eigh(mat: np.ndarray) -> np.ndarray:
    """Compute the eigenvalues and eigenvectors of a matrix.
    Returns in descending order with respect to eigenvalues.
    """
    # Returns orthonormal eigens
    w, v = np.linalg.eigh(mat)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]
    return w, v


def load_feret(datasets: dict) -> dict:
    """Load the FERET dataset into dictionary."""
    for dataset in datasets.values():
        for img_path in dataset["dir"].glob("*.pgm"):
            img = load_img(img_path)
            if img is not None:
                dataset["imgs"].append(img)
            else:
                msg = f"Failed to load image at {img_path}"
                raise ValueError(msg)
    print("Datasets loaded:")
    print({k: len(v["imgs"]) for k, v in datasets.items()})
    return datasets


def main() -> None:
    output_dir = pathlib.Path("attachments/train")
    output_dir.mkdir(exist_ok=True)
    datasets = {
        "fa_H": {"imgs": [], "dir": pathlib.Path("data/fa_H")},
        "fa2_H": {"imgs": [], "dir": pathlib.Path("data/fa_H")},
        "fa_L": {"imgs": [], "dir": pathlib.Path("data/fa_L")},
        "fa2_L": {"imgs": [], "dir": pathlib.Path("data/fa_L")},
        "fb_H": {"imgs": [], "dir": pathlib.Path("data/fb_H")},
        "fb_L": {"imgs": [], "dir": pathlib.Path("data/fb_L")},
    }
    datasets = load_feret(datasets)

    # Remove the first 50 subjects from fa2_H
    fa2_H_labels = []
    for img_path in datasets["fa2_H"]["dir"].glob("*.pgm"):
        person_id = img_path.stem.split("_")[0]
        fa2_H_labels.append(person_id)
    fa2_H_labels = np.array(fa2_H_labels)
    mask = fa2_H_labels.astype(int) > 140
    datasets["fa2_H"]["imgs"] = [img for img, m in zip(datasets["fa2_H"]["imgs"], mask) if m]
    print(f"Removed {np.sum(~mask)} images from fa2_H")

    # Remove the first 50 subjects from fa2_L
    fa2_L_labels = []
    for img_path in datasets["fa2_L"]["dir"].glob("*.pgm"):
        person_id = img_path.stem.split("_")[0]
        fa2_L_labels.append(person_id)
    fa2_L_labels = np.array(fa2_L_labels)
    mask = fa2_L_labels.astype(int) > 140
    datasets["fa2_L"]["imgs"] = [img for img, m in zip(datasets["fa2_L"]["imgs"], mask) if m]
    print(f"Removed {np.sum(~mask)} images from fa2_L")

    # Store average face for each dataset
    for name, dataset in datasets.items():
        avg_face = np.mean(dataset["imgs"], axis=0)
        datasets[name]["avg_face"] = avg_face

    # Flatten each image and calc it's mean-subtracted version
    for dataset in datasets.values():
        # Read person id from filename and store labels
        dataset["labels"] = []
        for img_path in dataset["dir"].glob("*.pgm"):
            person_id = img_path.stem.split("_")[0]
            dataset["labels"].append(person_id)

        dataset["flat_imgs"] = [img.flatten() for img in dataset["imgs"]]
        dataset["num_imgs"] = len(dataset["flat_imgs"])
        dataset["n_pixels"] = dataset["flat_imgs"][0].shape[0]
        dataset["flat_avg_face"] = dataset["avg_face"].flatten()
        dataset["mean_sub_imgs"] = [img - dataset["flat_avg_face"] for img in dataset["flat_imgs"]]

    # Calculate the small μ_i, v_i of A^T @ A
    # Then calculate the large λ_i, u_i of A @ A^T and normalize to unit
    for dataset in datasets.values():
        A = np.stack(dataset["mean_sub_imgs"], axis=1)
        cov = (A.T @ A) / dataset["num_imgs"]
        # Assert cov is symmetric
        assert np.allclose(cov, cov.T), "Covariance matrix is not symmetric"
        w, v = compute_eigh(cov)

        # Discard near-zero eigenvalues
        # otherwise, vectors break orthonormality
        # These are small because they are numerical rounding
        # errors
        thresh = 1e-8
        mask = w > thresh
        w, v = w[mask], v[:, mask]
        print(f"Discarded {np.sum(~mask)} near-zero eigenvalues")

        u = A @ v
        u = u / np.linalg.norm(u, axis=0)

        # Assert u is orthonormal
        assert np.allclose(u.T @ u, np.eye(u.shape[1])), "Eigenvectors are not orthonormal"

        # Assert Cu=λu
        # (A A^T)u = A(A^T u) to prevent creating the large matrix
        lhs = A @ (A.T @ u)
        lhs = lhs / dataset["num_imgs"]
        rhs = u * w
        assert np.allclose(lhs, rhs), "Eigenvectors do not satisfy Cu=λu"
        print(
            f"Computed {u.shape[1]} eigenfaces for dataset with {dataset['num_imgs']} images and {dataset['n_pixels']} pixels",
        )
        dataset["eigenvalues"] = w
        dataset["eigenvectors"] = u

        # Project each img onto the M Eigenfaces
        # Eigen-coefficient/projection
        dataset["projections"] = [img.T @ u for img in dataset["mean_sub_imgs"]]

    # Save dataset dict to file
    save_path = output_dir / "datasets.npy"
    np.save(save_path, datasets)

    # Test using Reconstruction
    for name, dataset in datasets.items():
        # Use mean subtracted img
        # Reconstruct with all k eigenfaces and add back the mean face
        i_hat = dataset["projections"][0] @ dataset["eigenvectors"].T + dataset["flat_avg_face"]
        # Compute distance from face space using euclidean
        dist = np.linalg.norm(i_hat - dataset["flat_imgs"][0])
        avg_dist = dist / dataset["n_pixels"]
        print(f"Average reconstruction error for {name}: {avg_dist:.2f}")

    for name, dataset in datasets.items():
        # Save to fig the average face
        ef = dataset["avg_face"]
        # Normalize to [0, 1]
        ef = (ef - ef.min()) / (ef.max() - ef.min())
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 1, 1)
        plt.imshow(ef, cmap="gray")
        plt.title("Average Face")
        plt.axis("off")
        plt.savefig(output_dir / f"{name}_average_face.png")
        plt.close()

        # and The eigenfaces corresponding to the 10 largest eigenvalues
        plt.figure(figsize=(20, 20))
        for i in range(10):
            plt.subplot(1, 10, i + 1)
            ef = dataset["eigenvectors"][:, i].reshape(dataset["avg_face"].shape)
            ef = (ef - ef.min()) / (ef.max() - ef.min())
            plt.imshow(
                ef,
                cmap="gray",
            )
            plt.title(f"Eigenface {i + 1}")
            plt.axis("off")
        plt.savefig(output_dir / f"{name}_eigenfaces.png")
        # The eigenfaces corresponding to the 10 smallest eigenvalues
        plt.figure(figsize=(20, 20))
        for i in range(10):
            plt.subplot(1, 10, i + 1)
            ef = dataset["eigenvectors"][:, -i - 1].reshape(dataset["avg_face"].shape)
            ef = (ef - ef.min()) / (ef.max() - ef.min())
            plt.imshow(
                ef,
                cmap="gray",
            )
            plt.title(f"Eigenface {dataset['num_imgs'] - i}")
            plt.axis("off")
        plt.savefig(output_dir / f"{name}_eigenfaces_small.png")
        plt.close()


if __name__ == "__main__":
    print("=== Training... ===")
    main()
