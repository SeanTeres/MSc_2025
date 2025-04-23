import numpy as np
import json
from datasets.hdf_dataset import HDF5Dataset
from utils import LABEL_SCHEMES, load_config


def save_split_indices(indices, file_path):
    """Save split indices to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(indices, f)


def get_label_scheme_supports(dataset, label_scheme):
    labels = dataset.get_labels(label_scheme)

    return np.array(labels)


def iter_strat(indexes, labels, split_size):
    n_samples = len(indexes)
    labels = np.array(labels)

    class_support = np.sum(labels, axis=0)

    class_ratios = class_support/n_samples * split_size

    desired_split = [int(cr) for cr in class_ratios]

    final_support_split = [0 for cr in class_ratios]

    created_split = []

    priority_labels = np.argsort(desired_split)

    assigned = np.array([False for i in range(n_samples)])

    while len(created_split) < split_size:
        current_priority = -1

        for pl in priority_labels:
            if desired_split[pl] > 0:
                candidate_indexes = (labels[:, pl] == 1) & (~assigned)
                if np.any(candidate_indexes):
                    current_priority = pl
                    break

        if current_priority == -1:
            remaining_indexes = np.where(~assigned)[0]
            random_index = np.random.choice(remaining_indexes)
            random_label = labels[random_index]
            final_support_split += random_label
            desired_split = desired_split - random_label
            created_split.append(random_index)
            assigned[random_index] = True
        else:
            first_viable = np.argmax(candidate_indexes)
            viable_label = labels[first_viable]
            created_split.append(indexes[first_viable])
            desired_split = desired_split - viable_label
            final_support_split += viable_label
            assigned[first_viable] = True

    return created_split


def stratify(dataset):
    """
    Visualize summary statistics and images from the dataset.

    Args:
        dataset: The HDF5 dataset to visualize.
    """

    prof_labels = get_label_scheme_supports(dataset, 'profusion_score')
    tb_labels = get_label_scheme_supports(dataset, 'tuberculosis')
    s_labels = [1 if x > 0 else 0 for x in prof_labels]
    tb_s_labels = np.logical_and(tb_labels, s_labels)

    _, prof_support = np.unique(prof_labels, return_counts=True)
    _, tb_support = np.unique(tb_labels, return_counts=True)
    _, s_support = np.unique(s_labels, return_counts=True)
    _, tb_s_support = np.unique(tb_s_labels, return_counts=True)

    indexes = [i for i in range(len(prof_labels))]
    labels = []

    for i in indexes:
        lab = [0, 0, 0, 0, 0, 0, 0]
        lab[prof_labels[i]] = 1
        lab[4] = 1 if tb_labels[i] == 1 else 0
        lab[5] = 1 if s_labels[i] == 1 else 0
        lab[6] = 1 if tb_s_labels[i] else 0
        labels.append(lab)

    print("=======TB==========")
    print(tb_support)

    print("=======SILICOSIS==========")
    print(s_support)

    print("=======PROFUSION SCORES==========")
    print(prof_support)

    print("======TB_SILICOSIS===========")
    print(tb_s_support)

    print("======OHE=======")
    print(np.sum(labels, axis=0))

    test_split = iter_strat(indexes, labels, 300)

    remaining_indexes = [i for i in indexes if i not in test_split]
    remaining_labels = [labels[i] for i in remaining_indexes]

    val_split = iter_strat(remaining_indexes, remaining_labels, 300)

    train_split = [i for i in remaining_indexes if i not in val_split]
    train_labels = [labels[i] for i in train_split]

    print(np.sum(train_labels, axis=0))

    print("================")
    print(train_split)
    print("================")
    print(val_split)
    print("================")
    print(test_split)

    train_split = [int(x) for x in train_split]
    val_split = [int(x) for x in val_split]
    test_split = [int(x) for x in test_split]

    save_split_indices(
        {"train": train_split,
         "val": val_split,
         "test": test_split},
        "stratified_split.json",
    )


if __name__ == "__main__":
    np.random.seed(42)
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

        stratify(dataset=dataset)

    except KeyError as e:
        print(f"Missing configuration: {e}")
    except FileNotFoundError as e:
        print(f"Dataset file not found: {e}")
