import numpy as np


def main(*, train: str, test: str, pc_info: float) -> None:

    datasets = np.load("attachments/datasets.npy", allow_pickle=True).item()
    train_ds = datasets[train]
    test_ds = datasets[test]

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
    train_ds["projections"] = train_ds["projections"][idx]

    # For each test image, project to train's eigenvector space
    test_ds["projected"] = np.dot(test_ds["mean_sub_imgs"], train_ds["eigenvectors"])

    correct = 0
    incorrect = 0

    # Loop through each projected img as the query
    for i, query in enumerate(train_ds["projections"]):
        # Mahalanobis distance
        closest_idx = np.argmin(
            np.sum((test_ds["projected"] - query) ** 2 / train_ds["eigenvalues"], axis=1),
        )

        # Check if closest_idx corresponds to the same class as query
        if test_ds["labels"][closest_idx] == test_ds["labels"][i]:
            correct += 1
        else:
            incorrect += 1

    print(
        f"Correct: {correct}, Incorrect: {incorrect}, Acc: {correct / (correct + incorrect):.2f}",
    )


if __name__ == "__main__":
    print("=== Testing... ===")

    train = "fa_H"
    test = "fb_H"
    # pc_info = float(input("Enter the amount of information to be preserved (i.e. 0.8): "))
    pc_info = 0.8
    if not (0 < pc_info <= 1):
        print("Invalid input. Please enter a number between 0 and 1.")
    main(train=train, test=test, pc_info=pc_info)
