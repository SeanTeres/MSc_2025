import sys
import os
import torch

# Add mbod-data-processor to the Python path
sys.path.append(os.path.abspath("../mbod-data-processor"))


import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from datasets.hdf_dataset import HDF5Dataset
from utils import LABEL_SCHEMES, load_config
from datasets.dataloader import get_dataloaders
from tsne import visualize_tsne, visualize_tsne2

def normalize(img, maxval=None, reshape=False):
    """Scales images to be roughly [-1024, 1024]."""
    if maxval is None:
        maxval = img.max()  # Dynamically detect max value

    img = (2 * (img.astype(np.float32) / maxval) - 1.) * 1024

    if reshape:
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        img = img[None, :, :]

    return img

preprocess = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor with values in [0, 1]
    transforms.Lambda(lambda x: torch.from_numpy(
        normalize(x.numpy(), maxval=1.0, reshape=True)
    )),  # Apply normalize function and convert back to tensor
])

if __name__ == "__main__":
    config = load_config()
    try:

        hdf5_file_path = config["merged_silicosis_output"]["hdf5_file"]
        ilo_hdf5_file_path = config["ilo_output"]["hdf5_file"]


        augmentations = transforms.Compose([
            transforms.RandomRotation(degrees=10, expand=False, fill=0),
            # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0)
        ])

        preprocess = transforms.Compose([
           # transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
           # transforms.Grayscale(),
            transforms.ToTensor(),
           # transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        mbod_dataset = HDF5Dataset(
            hdf5_path=hdf5_file_path,
            labels_key="tuberculosis",
            images_key="images",
            preprocess=preprocess,
            augmentations=None
        )


        train_loader, val_loader, test_loader = get_dataloaders(
            hdf5_path=hdf5_file_path,
            preprocess=None,
            batch_size=16,
            labels_key="tuberculosis",
            split_file="stratified_split.json",
            augmentations=None
        )

        kaggle_tb_dataset = HDF5Dataset(
            hdf5_path=config["kaggle_TB"]["outputpath"],
            labels_key="tuberculosis",
            images_key="images",
            preprocess=preprocess,
            augmentations=None
        )

        
        for i in range(5):
            label = kaggle_tb_dataset[i][1].numpy()
            image = kaggle_tb_dataset[i][0]

            print(f"Label: {label}, Image Type: {type(image)}, Shape: {image.shape}, Max: {image.max()}, Min: {image.min()}")

        plt.figure(figsize=(10, 10))


        # Display 4 images from each dataset

        plt.subplot(2, 4, 1)
        plt.imshow(kaggle_tb_dataset[0][0].numpy().squeeze(), cmap='gray')
        plt.title(f"Label: {kaggle_tb_dataset[0][1].numpy()}")

        plt.subplot(2, 4, 2)
        plt.imshow(kaggle_tb_dataset[1][0].numpy().squeeze(), cmap='gray')
        plt.title(f"Label: {kaggle_tb_dataset[1][1].numpy()}")

        plt.subplot(2, 4, 3)
        plt.imshow(kaggle_tb_dataset[2][0].numpy().squeeze(), cmap='gray')
        plt.title(f"Label: {kaggle_tb_dataset[2][1].numpy()}")
        plt.subplot(2, 4, 4)

        plt.imshow(kaggle_tb_dataset[3][0].numpy().squeeze(), cmap='gray')
        plt.title(f"Label: {kaggle_tb_dataset[3][1].numpy()}")
        plt.tight_layout()

        plt.subplot(2, 4, 5)
        plt.imshow(mbod_dataset[4][0].numpy().squeeze(), cmap='gray')
        plt.title(f"Label: {mbod_dataset[4][1].numpy()}")

        plt.subplot(2, 4, 6)
        plt.imshow(mbod_dataset[5][0].numpy().squeeze(), cmap='gray')
        plt.title(f"Label: {mbod_dataset[5][1].numpy()}")

        plt.subplot(2, 4, 7)
        plt.imshow(mbod_dataset[6][0].numpy().squeeze(), cmap='gray')
        plt.title(f"Label: {mbod_dataset[6][1].numpy()}")

        plt.subplot(2, 4, 8)
        plt.imshow(mbod_dataset[7][0].numpy().squeeze(), cmap='gray')
        plt.title(f"Label: {mbod_dataset[7][1].numpy()}")

        plt.tight_layout()

        plt.savefig('kaggle_samples.png')
        plt.show()

    except KeyError as e:
        print(f"Missing configuration: {e}")
