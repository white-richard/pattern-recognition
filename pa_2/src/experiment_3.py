from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from bayes import (
    est_sample_cov,
    est_sample_mean,
    maximum_gaus_value_c,
    mult_gaussian_discriminant,
)
from seed import set_all_seeds


def normalize_rgb_ycc(pixels: np.ndarray) -> np.ndarray:
    pass


def normalize_rgb_chromatic(pixels: np.ndarray) -> np.ndarray:
    """Normalize rgb pixels to chromatic space.

    R = R / (R + G + B)
    g = G / (R + G + B).
    """
    # python math needs fp32
    pixels = pixels.astype(np.float32)

    r = pixels[:, 0]
    g = pixels[:, 1]
    b = pixels[:, 2]

    psum = r + g + b

    # prevent division by zero
    # effectively sets the pixels to black in chromatic space
    # because {r,g} = 0/1 = 0
    psum[psum == 0] = 1

    r = r / psum
    g = g / psum

    return np.stack((r, g), axis=1)  # n x 2


def load_img(path: Path):
    """Load ppm image."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_ref_path(data_dir: Path, img_path: Path) -> Path:
    """Get the matching reference image path."""
    suffix = img_path.stem.replace("Training_", "")
    return data_dir / f"ref{suffix}.ppm"


def plot_imgs(img, ref_img) -> None:
    _fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(img)
    axes[0].axis("off")

    axes[1].imshow(ref_img)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def extract_faces_pixels(img, ref_img) -> np.ndarray:
    """Extract the pixels of the face from the image using the reference image."""
    face_mask = ground_truth_mask(ref_img)
    return img[face_mask]  # num_face_pixels x 3


def mle_estimate(img):
    """Calculate the MLE estimate of the mean and variance for each channel."""
    mean = np.mean(img, axis=(0, 1))
    var = np.var(img, axis=(0, 1))
    return mean, var


def estimate_face_model(img, ref_img, img_space: str):
    """Estimate the face model parameters from training data."""
    face_pixels = extract_faces_pixels(img, ref_img)
    if img_space == "chromatic":
        norm_face_pixels = normalize_rgb_chromatic(face_pixels)
    elif img_space == "ycc":
        norm_face_pixels = normalize_rgb_ycc(face_pixels)
    sample_mean = est_sample_mean(norm_face_pixels)
    sample_cov = est_sample_cov(norm_face_pixels, sample_mean)
    c = maximum_gaus_value_c(sample_cov)
    return sample_mean, sample_cov, c


def ground_truth_mask(ref_img) -> np.ndarray:
    """Calculate the ground truth mask from the reference image."""
    return np.any(ref_img > 0, axis=2)


def evaluate_thresholds(
    data_dir: Path,
    test_names: list[str],
    sample_mean: np.ndarray,
    sample_cov: np.ndarray,
    thresholds: np.ndarray,
    img_space: str,
) -> dict:
    """Evaluate thresholds over the test images."""
    metrics = {t: {"fp": 0, "fn": 0, "total_p": 0, "total_n": 0} for t in thresholds}

    def init_metrics(thresholds: np.ndarray) -> dict:
        """Initialize metric storage for each threshold."""
        return {t: {"fp": 0, "fn": 0, "total_p": 0, "total_n": 0} for t in thresholds}

    for test_name in test_names:
        test_path = data_dir / test_name
        ref_path = get_ref_path(data_dir, test_path)

        # load the images
        img = load_img(test_path)
        ref_img = load_img(ref_path)

        # flatten
        flat_img = img.reshape(-1, 3)
        # normalize
        if img_space == "chromatic":
            norm_flat_img = normalize_rgb_chromatic(flat_img)
        elif img_space == "ycc":
            norm_flat_img = normalize_rgb_ycc(flat_img)
        else:
            msg = f"Invalid image space: {img_space}"
            raise ValueError(msg)
        face_mask = ground_truth_mask(ref_img)

        # Discriminate
        likelihoods = mult_gaussian_discriminant(norm_flat_img, sample_mean, sample_cov)
        flat_face_mask = face_mask.flatten()

        # total positives are total number of face pixels
        total_pos = np.sum(flat_face_mask)
        total_neg = len(flat_face_mask) - total_pos

        for threshold in thresholds:
            pred = likelihoods > threshold

            # predicted true but was false
            fp = np.sum(pred & ~flat_face_mask)
            # predicted false but was true
            fn = np.sum(~pred & flat_face_mask)

            metrics[threshold]["fp"] += fp
            metrics[threshold]["fn"] += fn
            metrics[threshold]["total_p"] += total_pos
            metrics[threshold]["total_n"] += total_neg

    return metrics


def plot_roc_curve(avg_fpr: list[float], avg_fnr: list[float]) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(avg_fpr, avg_fnr, color="orange", linewidth=1, marker="s")
    plt.title("ROC Curve for Face Detection Performance.")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("False Negative Rate (FNR)")
    plt.grid(True)
    plt.show()


def main(img_space: str) -> None:
    set_all_seeds(42)
    data_dir = Path("Data_Prog2")

    train_path = data_dir / "Training_1.ppm"
    train_ref_path = get_ref_path(data_dir, train_path)

    train_img = load_img(train_path)
    train_ref_img = load_img(train_ref_path)
    plot_imgs(train_img, train_ref_img)

    sample_mean, sample_cov, c = estimate_face_model(train_img, train_ref_img, img_space=img_space)
    thresholds = np.linspace(0, c, 20)

    test_names = ["Training_3.ppm", "Training_6.ppm"]
    metrics = evaluate_thresholds(
        data_dir, test_names, sample_mean, sample_cov, thresholds, img_space=img_space
    )

    avg_fpr = []
    avg_fnr = []

    for t in thresholds:
        s = metrics[t]
        avg_fpr.append(s["fp"] / s["total_n"])
        avg_fnr.append(s["fn"] / s["total_p"])
    plot_roc_curve(avg_fpr, avg_fnr)

    # Find where the difference between fpr, fnr is smallest
    eer_idx = np.argmin(np.abs(np.array(avg_fpr) - np.array(avg_fnr)))
    best_threshold = thresholds[eer_idx]

    print(f"Equal Error Rate Threshold: {best_threshold:.6f}")
    print(f"FPR: {avg_fpr[eer_idx]:.4f}, FNR: {avg_fnr[eer_idx]:.4f}")


if __name__ == "__main__":
    img_space = "chromatic"  # chromatic | ycc
    main(img_space)
