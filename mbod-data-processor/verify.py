# Verify integrity, count class support
import numpy as np
from datasets.hdf_dataset import HDF5Dataset
from utils import load_config


def visualise(dataset):
    """
    Visualize summary statistics and images from the dataset.

    Args:
        dataset: The HDF5 dataset to visualize.
    """
    print("Dataset Summary:")
    print(dataset)

    # Extract the labels for the given label scheme
    labels = dataset.get_labels()

    # Compute support
    if labels.ndim == 1:  # Single-class labels
        unique_labels, support = np.unique(labels, return_counts=True)
    elif labels.ndim == 2:  # Multi-class/multi-label
        support = labels.sum(axis=0)
    else:
        print(f"Unsupported label dimensions: {labels.ndim}")
        return

    print(support)


if __name__ == "__main__":
    config = load_config()
    try:
        dataset_path = config["dataset_check"]["hdf5_file"]
        chosen_label_scheme = config["dataset_check"]["label_scheme"]

        dataset = HDF5Dataset(
            hdf5_path=dataset_path,
            labels_key=chosen_label_scheme,
            images_key="images",
            preprocess=None
        )

        visualise(dataset=dataset)

    except KeyError as e:
        print(f"Missing configuration: {e}")
    except FileNotFoundError as e:
        print(f"Dataset file not found: {e}")
