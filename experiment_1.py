import pathlib

import numpy as np

from bayes import discriminate_case_one
from generate_normals import generate_2d_gas_data

if __name__ == "__main__":
    fig_dir = pathlib.Path("attachments")

    # Class 1
    num_class1 = 60000
    mean_class1 = (1, 1)
    labels1 = np.full(num_class1, 1)

    # Class 2
    num_class2 = 140000
    mean_class2 = (4, 4)
    labels2 = np.full(num_class2, 2)

    std = 1
    total = num_class1 + num_class2

    # Calculate priors
    p1 = num_class1 / total
    p2 = num_class2 / total

    # Generate out normal points
    points1 = generate_2d_gas_data(num_class1, mean=mean_class1)
    points2 = generate_2d_gas_data(num_class2, mean=mean_class2)

    # Stack our points and labels for distrimination
    points = np.vstack((points1, points2))
    labels = np.concatenate((labels1, labels2))

    # Discriminate
    g1 = discriminate_case_one(x=points, p_i=p1, mean=mean_class1, std=std)
    g2 = discriminate_case_one(x=points, p_i=p2, mean=mean_class2, std=std)
    predictions = np.where(g1 > g2, 1, 2)

    # Error rate
    num_errors = np.sum(predictions != labels)
    error_rate = num_errors / total

    print(f"Total errors: {num_errors}")
    print(f"Minimum error rate: {error_rate * 100:.2f}%")
