# Visualize random X-rays, metadata, etc.
import matplotlib.pyplot as plt
import numpy as np
from datasets.hdf_dataset import HDF5SilicosisDataset
from utils import LABEL_SCHEMES, load_config


def visualise(dataset, num_images=6):
    """
    Visualize summary statistics and images from the dataset.

    Args:
        dataset: The HDF5 dataset to visualize.
        num_images: Number of random images to display.
    """
    label_names = dataset.get_label_names()

    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i, ax in enumerate(axes):
        idx = np.random.randint(0, len(dataset))
        sample = dataset[idx]
        image = sample["img"].numpy().squeeze()  # Convert tensor to numpy
        print(image.shape)
        label_data = sample["lab"].numpy()  # Get label data

        # Determine label names
        if np.isscalar(label_data) or label_data.ndim == 0:  # Single scalar value
            label_text = (
                label_names[int(label_data)]
                if int(label_data) < len(label_names)
                else "Unknown"
            )
        elif label_data.ndim == 1:  # Multi-label vector
            labels_with_names = [
                label_names[i] for i, active in enumerate(label_data) if active
            ]
            label_text = (
                ", ".join(labels_with_names) if labels_with_names else "No Finding"
            )
        else:
            label_text = "Unsupported label format"

        # Display the image with its labels
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        ax.set_title(label_text)
    plt.tight_layout()
    plt.savefig(f"visualise_{i}.png")


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

        visualise(dataset=dataset, num_images=6)

    except KeyError as e:
        print(f"Missing configuration: {e}")
    except FileNotFoundError as e:
        print(f"Dataset file not found: {e}")
