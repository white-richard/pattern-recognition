import pathlib

import cv2
import numpy as np


def load_img(path: pathlib.Path) -> np.ndarray | None:
    """Load ppm image."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compute_eigh(mat: np.ndarray, k: int) -> np.ndarray:
    """Compute the top k eigenvalues and eigenvectors of a matrix.
    Returns in descending order with respect to eigenvalues.
    """
    w, v = np.linalg.eig(mat)
    idx = np.argsort(w)[::-1][:k]
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


def calc_avg_face(dataset: dict) -> np.ndarray:
    """Calculate the average face of a dataset."""
    return np.mean(dataset["imgs"], axis=0)


def main() -> None:
    output_dir = pathlib.Path("attachments")
    output_dir.mkdir(exist_ok=True)
    datasets = {
        "fa_H": {"imgs": [], "dir": pathlib.Path("data/fa_H")},
        "fa_L": {"imgs": [], "dir": pathlib.Path("data/fa_L")},
        "fb_H": {"imgs": [], "dir": pathlib.Path("data/fb_H")},
        "fb_L": {"imgs": [], "dir": pathlib.Path("data/fb_L")},
    }
    datasets = load_feret(datasets)

    # Store average face for each dataset
    for name, dataset in datasets.items():
        avg_face = calc_avg_face(dataset)
        datasets[name]["avg_face"] = avg_face

    # Flatten each image and calc it's mean-subtracted version
    for dataset in datasets.values():
        dataset["flat_imgs"] = [img.flatten() for img in dataset["imgs"]]
        dataset["num_imgs"] = len(dataset["flat_imgs"])
        dataset["n_pixels"] = dataset["flat_imgs"][0].shape[0]
        dataset["flat_avg_face"] = dataset["avg_face"].flatten()
        dataset["mean_sub_imgs"] = [img - dataset["flat_avg_face"] for img in dataset["flat_imgs"]]

    # Calculate the small μ_i, v_i of A^T @ A
    # Then calculate the large λ_i, u_i of A @ A^T and normalize to unit
    for name, dataset in datasets.items():
        A = np.stack(dataset["mean_sub_imgs"], axis=1)
        cov = A.T @ A
        # Assert cov is symmetric
        assert np.allclose(cov, cov.T), "Covariance matrix is not symmetric"
        w, v = compute_eigh(cov, k=10)
        u = A @ v
        u = u / np.linalg.norm(u, axis=0)
        # Assert u is orthonormal
        assert np.allclose(u.T @ u, np.eye(u.shape[1])), "Eigenvectors are not orthonormal"
        # Assert Cu=λu
        assert np.allclose(A @ A.T @ u, u * w), "Eigenvectors do not satisfy Cu=λu"
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
        dist = np.linalg.norm(i_hat - dataset["flat_avg_face"])
        avg_dist = dist / dataset["n_pixels"]
        print(f"Average reconstruction error for {name}: {avg_dist:.2f}")


if __name__ == "__main__":
    print("=== Training... ===")
    main()
