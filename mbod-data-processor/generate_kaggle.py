from utils import load_config, save_to_h5py
from datasets.kaggle_tb import KaggleTBDataset
from utils import XRayResizer
import numpy as np

if __name__ == "__main__":
    config = load_config()

    try:
        data_path = config["kaggle_TB"]["path"]
        output_path = config["kaggle_TB"]["outputpath"]
        image_size = config["kaggle_TB"]["image_size"]

        dataset = KaggleTBDataset(datapath=data_path)

        num_samples = len(dataset)
        xrayresizer = XRayResizer(224)
        img_shape = xrayresizer(dataset[0]["img"])[0].shape

        hdf5_structure = {
            "images": {
                "shape": (num_samples,) + img_shape,
                "dtype": np.float32,
                "value_fn": lambda sample: xrayresizer(sample["img"])[0],
            },
            "tuberculosis": {"shape": (num_samples,), "dtype": np.int8},
        }

        save_to_h5py(
            dataset=dataset,
            filename=output_path,
            hdf5_structure=hdf5_structure,
            xrayresizer=xrayresizer,
        )
    except Exception as ex:
        print(ex)
