import torch
import pydicom
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader, Subset, ConcatDataset
import torchvision.transforms as transforms
import os
import torchxrayvision as xrv
from skimage.color import rgb2gray
from skimage.transform import resize
import pydicom
from torchxrayvision.datasets import XRayCenterCrop
import pandas as pd
import wandb
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, classification_report, confusion_matrix
import sys
sys.path.append('/home/sean/MSc/code')
import utils.helpers as helpers
import utils.classes as classes
import utils.train_utils as train_utils


# Log augmented images
def log_augmented_images(dataset, num_images=5):
    images = []
    for i in range(num_images):
        img, label = dataset[i]
        images.append(wandb.Image(img.permute(1, 2, 0).numpy(), caption=f"Label: {label}"))
    wandb.log({"Augmented Images": images})



with open('/home/sean/MSc/code/binary/clf_config.yaml', 'r') as file:
    config = yaml.safe_load(file)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dicom_dir_1 = '/home/sean/MSc/data/MBOD_Datasets/Dataset-1'
dicom_dir_2 = '/home/sean/MSc/data/MBOD_Datasets/Dataset-2'
metadata_1 = pd.read_excel("/home/sean/MSc/data/MBOD_Datasets/Dataset-1/FileDatabaseWithRadiology.xlsx")

metadata_2 = pd.read_excel("/home/sean/MSc/data/MBOD_Datasets/Dataset-2/Database_Training-2024.08.28.xlsx")

# target_label = 'Profusion and TBA-TBU'
# model_resolution = 224

# Initialize datasets
d1 = classes.DICOMDataset1(dicom_dir=dicom_dir_1, metadata_df=metadata_1)
d2 = classes.DICOMDataset2(dicom_dir=dicom_dir_2, metadata_df=metadata_2)

# Split datasets and store indices
train_indices_d1, val_indices_d1, test_indices_d1 = helpers.split_dataset(d1)
train_indices_d2, val_indices_d2, test_indices_d2 = helpers.split_dataset(d2)

# Save indices for later use
split_indices = {
    'd1': {'train': train_indices_d1, 'val': val_indices_d1, 'test': test_indices_d1},
    'd2': {'train': train_indices_d2, 'val': val_indices_d2, 'test': test_indices_d2}
}


for experiment_name, experiment in config['experiments'].items():
    print(f"Running experiment: {experiment_name}")
    lr = experiment['lr']
    n_epochs = experiment['n_epochs']
    batch_size = experiment['batch_size']
    train_dataset = experiment['train_dataset']
    model_name = experiment['model']
    oversampling = experiment['oversampling']
    loss_function = experiment['loss_function']
    augmentations = experiment['augmentation']
    model_resolution = experiment['model_resolution']
    target_label = experiment['target']

    # Create the transformation pipeline
    augmentations_list = transforms.RandomApply([
        # transforms.CenterCrop(np.round(224 * 0.9).astype(int)),  # Example crop
        transforms.RandomRotation(degrees=(-5, 5)),  
        transforms.Lambda(lambda img: helpers.salt_and_pepper_noise_tensor(img, prob=0.02)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.025, 0.025))
    ], p=0.5) 

    d1.set_target(target_label, model_resolution)
    d2.set_target(target_label, model_resolution)
    # Create augmented datasets
    d1_aug = classes.DICOMDataset1(dicom_dir=dicom_dir_1, metadata_df=metadata_1, transform=augmentations_list)
    d2_aug = classes.DICOMDataset2(dicom_dir=dicom_dir_2, metadata_df=metadata_2, transform=augmentations_list)

    d1_aug.set_target(target_label, model_resolution)
    d2_aug.set_target(target_label, model_resolution)

    

    train_d1 = Subset(d1, train_indices_d1)
    train_aug_d1 = Subset(d1_aug, train_indices_d1)

    val_d1 = Subset(d1, val_indices_d1)
    test_d1 = Subset(d1, test_indices_d1)

    train_d2 = Subset(d2, train_indices_d2)
    train_aug_d2 = Subset(d2_aug, train_indices_d2)

    val_d2 = Subset(d2, val_indices_d2)
    test_d2 = Subset(d2, test_indices_d2)

    if model_resolution == 224:
        print(f"Model is: {model_name}, model_resolution is: {model_resolution}")
        model = xrv.models.DenseNet(weights=model_name)
        model = model.to(device)

        in_features = 1024  # Based on the output of model.features2()
        model.classifier = classes.BaseClassifier(in_features).to(device)

    elif model_resolution == 512:
        print(f"Model is: {model_name}, model_resolution is: {model_resolution}")
        model = xrv.models.ResNet(weights=model_name)
        model = model.to(device)

        model.features2 = model.features
        in_features = 2048
        model.classifier = classes.BaseClassifier512(in_features).to(device)
    
    else:
        print("ERR: Unrecognized model resolution.")


    # Create dataloaders
    train_loader_d1, train_aug_loader_d1, val_loader_d1, test_loader_d1 = helpers.create_dataloaders(
        train_d1, train_aug_d1, val_d1, test_d1, experiment['batch_size'], target=target_label
    )

    train_loader_d2, train_aug_loader_d2, val_loader_d2, test_loader_d2 = helpers.create_dataloaders(
        train_d2, train_aug_d2, val_d2, test_d2, experiment['batch_size'], target=target_label
    )

    concat_train = ConcatDataset([train_d1, train_d2])

    concat_train_loader = DataLoader(concat_train, batch_size=experiment['batch_size'], shuffle=True)       

    # Print label distribution
    print("Label distribution for training set (D1):", helpers.calc_label_dist(d1, train_loader_d1.dataset, target_label + ' Label'))
    print("Label distribution for validation set (D1):", helpers.calc_label_dist(d1, val_loader_d1.dataset, target_label + ' Label'))
    print("Label distribution for test set (D1):", helpers.calc_label_dist(d1, test_loader_d1.dataset, target_label + ' Label'))
    print("*****"*50 + '\n')
    print("Label distribution for training set (D2):", helpers.calc_label_dist(d2, train_loader_d2.dataset, target_label + ' Label'))
    print("Label distribution for validation set (D2):", helpers.calc_label_dist(d2, val_loader_d2.dataset, target_label + ' Label'))
    print("Label distribution for test set (D2):", helpers.calc_label_dist(d2, test_loader_d2.dataset, target_label + ' Label'))
    print("*****"*50 + '\n')


    # Initialize wandb
    wandb.login(key = '176da722bd80e35dbc4a8cea0567d495b7307688')
    wandb.init(project='MBOD-New', name=experiment_name)
    wandb.config.update(experiment)
    # in_features = 1024  # Based on the output of model.features2()
    # model.classifier = classes.BaseClassifier(in_features).to(device)

    # Train and evaluate model
    test_labels_d1, test_labels_d2, test_preds_d1, test_preds_d2 = [], [], [], []

    if augmentations:
        print(f"ON THE FLY AUGMENTATION!")
        # Log augmented images from the training dataset
        log_augmented_images(train_aug_d1)
        train_loader_d1 = train_aug_loader_d1
        train_loader_d2 = train_aug_loader_d2
    else:
        print(f"NO AUGMENTATION!")
        train_loader_d1 = train_loader_d1
        train_loader_d2 = train_loader_d2

    if train_dataset == "MBOD 1":
        print("Training on Dataset 1\n")

        if(loss_function == "CrossEntropyLoss"):
            if(oversampling):
                pos_weight = helpers.compute_pos_weight(train_d1, target_label + ' Label')
                wandb.log({
                    "BCE pos_weight": pos_weight
                })

                print(f"Oversampling with pos_weight = {pos_weight} ---- dataset {train_dataset}")
            else:
                pos_weight = torch.tensor([1.0])
                wandb.log({
                    "BCE pos_weight": pos_weight
                })
            
            model = train_utils.train_model(train_loader_d1, val_loader_d1, model, n_epochs, lr, device, pos_weight=pos_weight, experiment_name=experiment_name)


        elif (loss_function == "FocalLoss"):

            alpha_d1 = helpers.get_alpha_FLoss(train_d1, target_label + ' Label')
            gamma = 2
            wandb.log({
                "FLoss alpha": alpha_d1,
                "FLoss gamma": gamma
            })
            print(f"Focal Loss with alpha = {alpha_d1} ---- dataset {train_dataset}")


            model = train_utils.train_model_with_focal_loss(train_loader_d1, val_loader_d1, model, n_epochs, lr, device, alpha=alpha_d1, gamma=gamma, experiment_name=experiment_name)
            
        else:
            print("ERR: Loss function must be CrossEntropyLoss or FocalLoss.")

    elif(train_dataset == "MBOD 2"):
        print("Training on Dataset 2\n")

        if(loss_function == "CrossEntropyLoss"):

            if(oversampling):
                pos_weight = helpers.compute_pos_weight(train_d2, target_label + ' Label')
                wandb.log({
                    "BCE pos_weight": pos_weight
                })
                print(f"Oversampling with pos_weight = {pos_weight} ---- dataset {train_dataset}")

            else:
                pos_weight = torch.tensor([1.0])
                wandb.log({
                    "BCE pos_weight": pos_weight
                })

            model = train_utils.train_model(train_loader_d2, val_loader_d2, model, n_epochs, lr, device, pos_weight=pos_weight, experiment_name=experiment_name)

        elif (loss_function == "FocalLoss"):
            alpha_d2 = helpers.get_alpha_FLoss(train_d2, target_label + ' Label')
            gamma = 2
            print(f"Focal Loss with alpha = {alpha_d2} ---- dataset {train_dataset}")

            wandb.log({
                "FLoss alpha": alpha_d2,
                "FLoss gamma": gamma
            })



            model = train_utils.train_model_with_focal_loss(train_loader_d2, val_loader_d2, model,
                                                            n_epochs, lr, device, alpha=alpha_d2, gamma=gamma, experiment_name=experiment_name)
    
    elif(train_dataset == "Combined"):
        print("Training on Combined Dataset\n")

        if(loss_function == "CrossEntropyLoss"):
            if(oversampling):
                pos_weight_d1 = helpers.compute_pos_weight(train_d1, target_label + ' Label')
                pos_weight_d2 = helpers.compute_pos_weight(train_d2, target_label + ' Label')
                pos_weight = (pos_weight_d1 + pos_weight_d2) / 2
                wandb.log({
                    "BCE pos_weight": pos_weight
                })
                print(f"Oversampling with pos_weight = {pos_weight} ---- dataset {train_dataset}")

            else:
                pos_weight = torch.tensor([1.0])
                wandb.log({
                    "BCE pos_weight": pos_weight
                })

            model = train_utils.train_model(concat_train_loader, val_loader_d2, model, n_epochs, lr, device, pos_weight=pos_weight, experiment_name=experiment_name)

        elif (loss_function == "FocalLoss"):
            alpha_d1 = helpers.get_alpha_FLoss(train_d1, target_label + ' Label')
            alpha_d2 = helpers.get_alpha_FLoss(train_d2, target_label + ' Label')
            alpha = (alpha_d1 + alpha_d2) / 2
            gamma = 2
            print(f"Focal Loss with alpha = {alpha} ---- dataset {train_dataset}")

            wandb.log({
                "FLoss alpha": alpha,
                "FLoss gamma": gamma
            })

            model = train_utils.train_model_with_focal_loss(concat_train_loader, val_loader_d1, model, n_epochs, lr, device, alpha=alpha, gamma=gamma, experiment_name=experiment_name)


    else:
        print("ERR: Unrecognized dataset name.")

    if(loss_function == 'CrossEntropyLoss'):

        test_labels_d1, test_preds_d1 = train_utils.test_model(test_loader_d1, model, device, "MBOD 1")
        test_labels_d2, test_preds_d2 = train_utils.test_model(test_loader_d2, model, device, "MBOD 2")

    elif(loss_function == 'FocalLoss'):
        alpha_d1 = helpers.get_alpha_FLoss(train_d1, target_label + ' Label')
        alpha_d2 = helpers.get_alpha_FLoss(train_d2, target_label + ' Label')
        print(f"testing Focal Loss with alpha = {alpha_d1} ---- MBOD 1")
        print(f"testing Focal Loss with alpha = {alpha_d2} ---- MBOD 2")

        test_labels_d1, test_preds_d1 = train_utils.test_model_with_focal_loss(test_loader_d1, model, device, "MBOD 1", alpha=alpha_d1, gamma=gamma)
        test_labels_d2, test_preds_d2 = train_utils.test_model_with_focal_loss(test_loader_d2, model, device, "MBOD 2", alpha=alpha_d2, gamma=gamma)
    # Generate and save confusion matrices
    report_d1 = classification_report(test_labels_d1, test_preds_d1)
    cm_d1 = confusion_matrix(test_labels_d1, test_preds_d1)

    report_d2 = classification_report(test_labels_d2, test_preds_d2)
    cm_d2 = confusion_matrix(test_labels_d2, test_preds_d2)

    print(f"Classification Report ({experiment_name}- MBOD 1):")
    print(report_d1)

    print(f"Confusion Matrix ({experiment_name}- MBOD 1):")
    print(cm_d1)

    print("*" * 50)

    print(f"Classification Report ({experiment_name} - MBOD 2)")
    print(report_d2)

    print(f"{experiment_name} MBOD 2")
    print(cm_d2)

    

    plt.figure(figsize=(14, 8))

    plt.subplot(1, 2, 1)
    sns.heatmap(cm_d1, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{experiment_name} - D1")

    plt.subplot(1, 2, 2)
    sns.heatmap(cm_d2, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{experiment_name} - D2")

    wandb.log({f"conf_mat": wandb.Image(plt)})

    combined_cm_d1 = helpers.plot_combined_conf_mat('Profusion', d1, test_preds_d1, test_indices_d1, True, "MBOD 1")
    combined_cm_d2 = helpers.plot_combined_conf_mat('Profusion', d2, test_preds_d2, test_indices_d2, True, "MBOD 2")

    
    print(f"Test preds: {test_preds_d2}")
    print(f"Test labels: {test_labels_d2}")

    wandb.finish()