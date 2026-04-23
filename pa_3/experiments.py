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
    np.argsort(w)[::-1]
    w = w[k]
    v = v[:, k]
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
        dataset["flat_avg_face"] = dataset["avg_face"].flatten()
        dataset["mean_sub_imgs"] = [img - dataset["flat_avg_face"] for img in dataset["flat_imgs"]]

    # Calculate the small μ_i, v_i of A^T @ A
    # Then calculate the large λ_i, u_i of A @ A^T and normalize to unit
    for dataset in datasets.values():
        A = np.stack(dataset["mean_sub_imgs"], axis=1)
        cov = A.T @ A
        w, v = compute_eigh(cov, k=10)
        u = A @ v
        u = u / np.linalg.norm(u, axis=0)
        dataset["eigenvalues"] = w
        dataset["eigenvectors"] = u


if __name__ == "__main__":
    main()
