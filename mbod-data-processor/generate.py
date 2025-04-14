from datasets.mbod_silicosis_dataset import MBODSilicosisDataset
from datasets.merge_dataset import MergeDataset
from utils import save_to_h5py, load_config
from utils import XRayResizer
import h5py
import numpy as np

if __name__ == "__main__":
    config = load_config()

    image_size = config["merged_silicosis_output"]["image_size"]

    try:
        d_silicosis_857 = MBODSilicosisDataset(
            imgpath=config["silicosis_857"]["imgpath"],
            csvpath=config["silicosis_857"]["csvpath"],
            delimeter=config["silicosis_857"]["delimeter"],
            image_id_column=config["silicosis_857"]["image_id_column"],
            radiologist_findings_columns=config["silicosis_857"][
                "radiologist_findings_columns"
            ],
            profusion_score_column=config["silicosis_857"]["profusion_score_column"],
        )
        d_silicosis_1179 = MBODSilicosisDataset(
            imgpath=config["silicosis_1179"]["imgpath"],
            csvpath=config["silicosis_1179"]["csvpath"],
        )

        d_silicosis_merge = MergeDataset([d_silicosis_857, d_silicosis_1179])
        # d_silicosis_merge = d_silicosis_v2

        # Define the structure of the HDF5 dataset
        num_samples = len(d_silicosis_merge)
        xrayresizer = XRayResizer(image_size)
        img_shape = xrayresizer(d_silicosis_merge[0]["img"])[0].shape
        hdf5_structure = {
            "images": {
                "shape": (num_samples,) + img_shape,
                "dtype": np.float32,
                "value_fn": lambda sample: xrayresizer(sample["img"])[0],
            },
            "lab": {
                "shape": (num_samples, len(d_silicosis_merge.pathologies)),
                "dtype": np.int8,
            },
            "study_id": {"shape": (num_samples,), "dtype": h5py.string_dtype("utf-8")},
            "tuberculosis": {"shape": (num_samples,), "dtype": np.int8},
            "silicosis": {"shape": (num_samples,), "dtype": np.int8},
            "silicosis_tuberculosis": {"shape": (num_samples,), "dtype": np.int8},
            "active_tuberculosis": {"shape": (num_samples,), "dtype": np.int8},
            "full_tuberculosis": {"shape": (num_samples, 4), "dtype": np.int8},
            "profusion_score": {"shape": (num_samples,), "dtype": np.int8},
        }

        save_to_h5py(
            dataset=d_silicosis_merge,
            hdf5_structure=hdf5_structure,
            filename=config["merged_silicosis_output"]["hdf5_file"],
            xrayresizer=xrayresizer,
        )
    except KeyError as e:
        print(f"Missing configuration: {e}")
