from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def load_img(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> None:
    data_dir = Path("Data_Prog2")
    training_files = sorted(data_dir.glob("Training_*.ppm"))

    for train_path in training_files:
        suffix = train_path.stem.replace("Training_", "")
        ref_path = data_dir / f"ref{suffix}.ppm"

        train_img = load_img(train_path)
        ref_img = load_img(ref_path)

        if train_img is None or ref_img is None:
            msg = f"Failed to load: {train_path.name}"
            raise ValueError(msg)

        _fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].imshow(train_img)
        axes[0].set_title(train_path.name)
        axes[0].axis("off")

        axes[1].imshow(ref_img)
        axes[1].set_title(ref_path.name)
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
