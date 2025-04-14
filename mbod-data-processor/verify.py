# Verify integrity, count class support
import matplotlib.pyplot as plt
import numpy as np
from datasets.hdf_dataset import HDF5SilicosisDataset
from utils import LABEL_SCHEMES, load_config


def visualise(dataset):
    """
    Visualize summary statistics and images from the dataset.

    Args:
        dataset: The HDF5 dataset to visualize.
    """
    print("Dataset Summary:")
    print(dataset)

    label_scheme = dataset.labels_key

    # Check if the label scheme exists in the dataset
    if label_scheme not in dataset.hdf5_file:
        print(f"Label scheme '{label_scheme}' not found in the dataset.")
        return

    # Extract the labels for the given label scheme
    labels = dataset.hdf5_file[label_scheme][:]
    label_names = dataset.get_label_names()

    # Compute support
    if labels.ndim == 1:  # Single-class labels
        unique_labels, support = np.unique(labels, return_counts=True)
        label_names = [label_names[i] for i in unique_labels]
    elif labels.ndim == 2:  # Multi-class/multi-label
        support = labels.sum(axis=0)
    else:
        print(f"Unsupported label dimensions: {labels.ndim}")
        return

    # Plot support
    plt.figure(figsize=(12, 6))
    plt.bar(label_names, support, color="skyblue")
    plt.xlabel("Labels")
    plt.ylabel("Support (Number of Samples)")
    plt.title(f"Support per Class ({label_scheme})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("verify.png")


if __name__ == "__main__":
    config = load_config()
    try:
        dataset_path = config["dataset_check"]["hdf5_file"]
        chosen_label_scheme = config["dataset_check"]["label_scheme"]

        dataset = HDF5SilicosisDataset(
            hdf5_file_path=dataset_path,
            labels_key=chosen_label_scheme,
            image_key="images",
            label_metadata=LABEL_SCHEMES,
        )

        visualise(dataset=dataset)

    except KeyError as e:
        print(f"Missing configuration: {e}")
    except FileNotFoundError as e:
        print(f"Dataset file not found: {e}")
