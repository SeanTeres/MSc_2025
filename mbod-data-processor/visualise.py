import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from datasets.hdf_dataset import HDF5Dataset
from utils import LABEL_SCHEMES, load_config


def visualise(dataset, num_images=6):
    """
    Visualize summary statistics and images from the dataset.

    Args:
        dataset: The HDF5 dataset to visualize.
        num_images: Number of random images to display.
    """
    label_names = LABEL_SCHEMES[dataset.labels_key]

    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i, ax in enumerate(axes):
        idx = np.random.randint(0, len(dataset))
        sample = dataset[idx]
        image = sample[0].numpy().squeeze()
        label_data = sample[1].numpy()

        if np.isscalar(label_data) or label_data.ndim == 0:
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

        augmentations = transforms.Compose([
            transforms.RandomRotation(degrees=10, expand=False, fill=0),
            # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0)
        ])

        preprocess = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.Grayscale(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        dataset = HDF5Dataset(
            hdf5_path=dataset_path,
            labels_key=chosen_label_scheme,
            images_key="images",
            preprocess=preprocess,
            augmentations=augmentations
        )

        visualise(dataset=dataset, num_images=6)

    except KeyError as e:
        print(f"Missing configuration: {e}")
    except FileNotFoundError as e:
        print(f"Dataset file not found: {e}")
