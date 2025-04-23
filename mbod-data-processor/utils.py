import sys
import numpy as np
import warnings
import skimage
import h5py
from tqdm import tqdm
import yaml
import os

LABEL_SCHEMES = {
    "lab": [
        "Silicosis not present.",  # 0/0
        "Possibility that Silicosis present considered but rejected.",  # 0/1
        "Severity 1: Small opacities present (refer to standard films).",  # 1
        "Silicosis present — insufficient small opacities to classify as category 1.",  # 1/0
        "Severity 2: Small opacities present (refer to standard films).",  # 2
        "Severity 3: Small opacities present (refer to standard films).",  # 3
        "Large opacity present with greatest diameter between 1 cm and 5 cm.",  # A
        "Large opacity or opacities present with combined area < right upper lobe.",  # B
        "One or more large opacities with a combined area > right upper lobe.",  # C
        "Cerebellopontine angle?",  # cpa
        "Cavitation",  # cv
        "Effusion",  # ef
        "Emphysema",  # em
        "Egg-shell calcification of glands",  # es
        "Enlargement of hilar or mediastinal glands",  # hi
        "Heart and vessels",  # hv
        "Other pathology or abnormality not represented.",  # oth
        "Small, rounded opacities < 1.5 mm in diameter.",  # p
        "Costophrenic angle obliteration",  # pla
        "Pleural calcification (plaque)",  # plc
        "Pleural wall abnormality — diffuse thickening",  # plw
        "Small, rounded opacities 1.5 to 3 mm in diameter.",  # q
        "Small, rounded opacities 3 to 10 mm in diameter.",  # r
        "Small, irregular opacities < 1.5 mm in width.",  # s
        "Small, irregular opacities 1.5 to 3 mm in width.",  # t
        "Active tuberculosis, probably",  # tba
        "Tuberculosis — activity uncertain",  # tbu
        "Small, irregular opacities 3 to 10 mm in width.",  # u
    ],
    "tuberculosis": ["No TB", "TB"],
    "silicosis": ["No Silicosis", "Silicosis"],
    "silicosis_tuberculosis": ["No ST", "ST"],
    "active_tuberculosis": ["Not Active TB (Includes Healthy)", "Active TB"],
    "full_tuberculosis": [
        "Active Tuberculosis",
        "Uncertain Tuberculosis",
        "Unhealthy Non-Tuberculosis",
        "Healthy/No findings",
    ],
    "profusion_score": [
        "0",
        "1",
        "2",
        "3"
    ],
}


class XRayResizer(object):
    """Resize an image to a specific size"""

    def __init__(self, size: int, engine="skimage"):
        self.size = size
        self.engine = engine
        if "cv2" in sys.modules:
            print("Setting XRayResizer engine to cv2 could increase performance.")

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.engine == "skimage":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return skimage.transform.resize(
                    img, (1, self.size, self.size), mode="constant", preserve_range=True
                ).astype(np.float32)
        elif self.engine == "cv2":
            import cv2  # pip install opencv-python

            return (
                cv2.resize(
                    img[0, :, :], (self.size, self.size), interpolation=cv2.INTER_AREA
                )
                .reshape(1, self.size, self.size)
                .astype(np.float32)
            )
        else:
            raise Exception("Unknown engine, Must be skimage (default) or cv2.")


def save_to_h5py(dataset, filename, hdf5_structure, xrayresizer):
    output_file = filename
    num_samples = len(dataset)

    with h5py.File(output_file, "w") as hdf:
        datasets = {
            key: hdf.create_dataset(key, shape=config["shape"], dtype=config["dtype"])
            for key, config in hdf5_structure.items()
        }

        for i in tqdm(range(num_samples), desc="Saving to HDF5"):
            sample = dataset[i]
            for key, config in hdf5_structure.items():
                value = config.get("value_fn", lambda s: s[key])(sample)
                datasets[key][i] = value

    print(f"Dataset saved to {output_file}")


def load_config(config_path="config.yaml"):
    """
    Load the YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.
    Returns:
        A dictionary containing the configuration settings.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    with open(config_path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error reading the configuration file: {e}")
