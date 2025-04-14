import os
from PIL import Image
from tqdm import tqdm
from utils import load_config
from dataloder import get_dataloaders
import numpy as np
import csv
from sklearn.model_selection import train_test_split

target_path = "/home/joshua/Documents/BenchX/datasets/MBOD_Silicosis"


def process_data(loader, directory, txtpath, allcsvpath):
    with open(txtpath, 'a+') as txtfile, open(allcsvpath, 'a+', newline='') as allcsvfile:

        allcsv_writer = csv.writer(allcsvfile)

        for i, item in tqdm(enumerate(loader)):
            image, label, image_id = item['img'].numpy().squeeze(), item['lab'].numpy()[0], item['image_id'][0]

            if label == 1.0 or label == 1:
                label = "Positive"
            else:
                label = "Negative"

            min_val, max_val = np.min(image), np.max(image)
            image = ((image - min_val) / (max_val - min_val)) * 255
            image = image.clip(0, 255).astype(np.uint8)

            image_pil = Image.fromarray(image, mode="L").convert("RGB")

            img_filename = f"{image_id}.png"
            img_directory = os.path.join(directory, img_filename)
            image_pil.save(img_directory, "PNG")

            txtfile.write(image_id + '\n')

            allcsv_writer.writerow([image_id, image_id, label])


def split_train_dataset(seed, traintxtpath):
    with open(traintxtpath, 'r') as file:
        files = [line.strip() for line in file.readlines()]

    x_train_1, _ = train_test_split(files, test_size=0.99, random_state=seed)
    x_train_10, _ = train_test_split(files, test_size=0.90, random_state=seed)

    train1path = os.path.join(target_path, "train_1.txt")
    train10path = os.path.join(target_path, "train_10.txt")

    with open(train1path, 'a+', newline='') as train1file, open(train10path, 'a+') as train10file:
        for item in x_train_1:
            train1file.write(item + "\n")

        for item in x_train_10:
            train10file.write(item + "\n")


if __name__ == "__main__":
    config = load_config()

    dataset_path = config["dataset_check"]["hdf5_file"]
    chosen_label_scheme = config["dataset_check"]["label_scheme"]
    all_csv_path = os.path.join(target_path, "mbod_silicosis_labels.csv")

    train_loader, val_loader, test_loader = get_dataloaders(dataset_path, train_split=0.6, batch_size=1, labels_key=chosen_label_scheme)

    data_split = ['train', 'val', 'test']

    with open(all_csv_path, 'a+', newline='') as allcsvfile:
        writer = csv.writer(allcsvfile)
        writer.writerow(["image_id", "rad_id", "class_name"])

    for split in data_split:
        output_image_dir = os.path.join(target_path, "images")
        txtpath = os.path.join(target_path, f"{split}.txt")
        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)

        if split == 'train':
            process_data(train_loader, output_image_dir, txtpath, all_csv_path)
            split_train_dataset(42, txtpath)
        elif split == 'val':
            process_data(val_loader, output_image_dir, txtpath, all_csv_path)
        elif split == 'test':
            process_data(test_loader, output_image_dir, txtpath, all_csv_path)
