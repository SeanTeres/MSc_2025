from datasets.hdf_dataset import HDF5SilicosisDataset
from utils import LABEL_SCHEMES, load_config
from data_splits import stratify, get_label_scheme_supports
import numpy as np
import matplotlib.pyplot as plt
import h5py
from dataloader import get_dataloaders, get_stratified_dataloaders
import torchxrayvision as xrv
import torch
from train_utils import classes, helpers
import torch.nn.functional as F
import torch.nn as nn
import wandb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, cohen_kappa_score, roc_auc_score
import seaborn as sns
from sklearn.calibration import calibration_curve
import io
import torchvision.transforms as transforms
import os

def check_empty_study_ids(hdf5_path):
    """
    Check how many samples have an empty study_id column
    
    Args:
        hdf5_path: Path to the HDF5 file
    """
    with h5py.File(hdf5_path, "r") as f:
        total_samples = f["study_id"].shape[0]
        empty_count = 0
        problematic_indices = []
        
        print(f"Checking {total_samples} study IDs for empty values...")
        
        for idx in range(total_samples):
            study_id = f["study_id"][idx]
            if isinstance(study_id, bytes):
                study_id = study_id.decode('utf-8')
            
            # Check if study_id is empty or just whitespace
            if not study_id or study_id.strip() == '':
                empty_count += 1
                problematic_indices.append(idx)
                
        print(f"\nFound {empty_count} empty study IDs out of {total_samples} samples ({empty_count/total_samples:.2%})")
        
        if empty_count > 0 and empty_count <= 10:
            print("\nIndices with empty study IDs:")
            for idx in problematic_indices:
                print(f"  Sample {idx}")
        elif empty_count > 10:
            print("\nFirst 10 indices with empty study IDs:")
            for idx in problematic_indices[:10]:
                print(f"  Sample {idx}")
                
        # Also check for study IDs that don't have enough parts when split by '.'
        problem_format_count = 0
        for idx in range(total_samples):
            study_id = f["study_id"][idx]
            if isinstance(study_id, bytes):
                study_id = study_id.decode('utf-8')
            
            parts = study_id.split('.')
            if len(parts) < 3:
                problem_format_count += 1
                
        print(f"\nFound {problem_format_count} study IDs without at least 3 parts when split by '.' ({problem_format_count/total_samples:.2%})")


def explore_hdf5_dataset(hdf5_path, num_samples=5):
    """
    Explore and display various fields from an HDF5 dataset
    
    Args:
        hdf5_path: Path to the HDF5 file
        num_samples: Number of samples to display (default: 5)
    """
    with h5py.File(hdf5_path, "r") as f:
        # Print available fields
        print("Available fields in the dataset:")
        for key in f.keys():
            print(f"- {key}: shape {f[key].shape}, dtype {f[key].dtype}")
        
        # Get total number of samples
        total_samples = f["images"].shape[0]
        print(f"\nTotal number of samples: {total_samples}")
        
        # Select random indices to explore
        indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)
        
        print(f"\nExploring {len(indices)} random samples:")
        
        for idx in indices:
            print(f"\nSample {idx}:")
            
            # Study ID
            study_id = f["study_id"][idx]
            if isinstance(study_id, bytes):
                study_id = study_id.decode('utf-8')
            print(f"  Study ID: {study_id}")
            
            # Labels
            print(f"  Lab: {f['lab'][idx]}")
            
            # Disease flags
            print(f"  Tuberculosis: {f['tuberculosis'][idx]}")
            print(f"  Silicosis: {f['silicosis'][idx]}")
            print(f"  Silicosis+TB: {f['silicosis_tuberculosis'][idx]}")
            print(f"  Active TB: {f['active_tuberculosis'][idx]}")
            print(f"  Full TB: {f['full_tuberculosis'][idx]}")
            print(f"  Profusion Score: {f['profusion_score'][idx]}")

            print(f"Shape: {f['images'][idx].shape}")
            print(f"min, max: {f['images'][idx].min()}, {f['images'][idx].max()}")
            
            # Display the image
            # plt.figure(figsize=(5, 5))
            # plt.imshow(f["images"][idx], cmap='gray')
            # plt.title(f"Sample {idx} - Profusion: {f['profusion_score'][idx]}")
            # plt.axis('off')
            # plt.show()



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("*" * 50)
    print(f"Using device: {device}")
    print("*" * 50)
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    config = load_config()
    
    try:
        # Get the path to the generated HDF5 file
        hdf5_file_path = config["merged_silicosis_output"]["hdf5_file"]
        

        check_empty_study_ids(hdf5_file_path)


        # Create an HDF5SilicosisDataset instance
        mbod_dataset_merged = HDF5SilicosisDataset(
            hdf5_file_path=hdf5_file_path,
            labels_key="lab",  # Main pathology labels, 'lab' for all labels
            image_key="images",
            label_metadata=LABEL_SCHEMES,
            data_aug=None,
            transform=None
        )

        augmentations_list = [
            # transforms.CenterCrop(np.round(224 * 0.9).astype(int)),  # Example crop
            transforms.RandomRotation(degrees=(-5, 5)),  
            transforms.Lambda(lambda img: helpers.salt_and_pepper_noise_tensor(img, prob=0.02)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.025, 0.025))
        ]


        ilo_imgs = classes.ILOImagesDataset(
            "C:/Users/user-pc/Masters/MSc_2025/data/ilo-radiographs-dicom",
            target_size=512,
            transform=None,
            filter_one_per_label=False,
            augment_label_0=False
        )

        img, label, study_id = ilo_imgs[7]

        label_to_indices = {}
        for i in range(len(ilo_imgs)):
            _, ilo_label, _ = ilo_imgs[i]
            if ilo_label not in label_to_indices:
                label_to_indices[ilo_label] = []
            label_to_indices[ilo_label].append(i)
            
        wandb.login(key = '176da722bd80e35dbc4a8cea0567d495b7307688')
        wandb.init(project='MBOD-cl',
           config={
               "batch_size": 16,
               "n_epochs": 20,
               "margin": 1.0,
                "learning_rate": 0.001
           })    
            
        model = xrv.models.ResNet(weights="resnet50-res512-all")
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        margin = 1.0
        triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

        n_epochs = wandb.config.n_epochs
        batch_size = wandb.config.batch_size

        # Get the dataloaders
        train_loader, val_loader, test_loader = get_stratified_dataloaders(
            hdf5_file_path,
            batch_size=wandb.config.batch_size,
            labels_key="profusion_score",
            image_key="images",
            oversample=False
        )


        model.train()

        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}/{n_epochs}")
            print("=" * 50)
            running_loss = 0.0

            all_embeddings = []
            all_labels = [] 
            for batch_idx, sample in enumerate(train_loader):
                optimizer.zero_grad()

                # Get batch and send to device.
                # img: shape [batch_size, channels, height, width]
                img = sample["img"].unsqueeze(1).to(device)  # Assuming img is single-channel and unsqueeze adds the channel dim.
                lab = sample["lab"].long().to(device)
                study_id = sample["image_id"]

                # Print some batch info (optional)

                if (batch_idx + 1) % 20 == 0:
                    print(f"Batch {batch_idx + 1}:")
                    print("img shape:", img.shape)
                    print("Labels:", lab)
                    print("Study IDs:", study_id)
                    print("=" * 50)

                # Extract features from current batch.
                # Here, model.features(img) returns a tensor of shape [batch_size, C, H, W]
                features = model.features(img)  

                # Normalize embeddings:
                embeddings = F.normalize(features, p=2, dim=1)  # [batch_size, C]
                all_embeddings.append(embeddings.detach().cpu())
                all_labels.append(lab.detach().cpu())

                # Get labels from the batch (on CPU for indexing is fine; you can always convert back)
                current_batch_labels = lab.cpu().numpy()

                # Accumulators for loss over the batch
                batch_triplet_loss = 0.0
                n_triplets = 0

                # For each sample in the batch, build a triplet:
                for i, positive_label in enumerate(current_batch_labels):
                    # Positive embedding (from the current batch):
                    positive_embedding = embeddings[i].unsqueeze(0)  # shape [1, C]

                    # Select an anchor from ILO images with the same label:
                    possible_anchors = {key: value for key, value in label_to_indices.items() if key == positive_label}
                    # Flatten the list of indices from the dictionary (if exists)
                    anchors = [idx for indices in possible_anchors.values() for idx in indices]
                    if len(anchors) == 0:
                        # If no matching anchor is found, skip this triplet.
                        continue

                    anchor_idx = np.random.choice(anchors)
                    # Get the anchor image from your ILO dataset.
                    anchor_img, anchor_label, anchor_study_id = ilo_imgs[anchor_idx]

                    # Process the anchor image: convert it to tensor, add batch dim, and send to device.
                    # (Adjust preprocessing as necessary.)
                    anchor_img_tensor = anchor_img.unsqueeze(0).to(device)  # e.g. shape [1, 1, H, W]
                    anchor_features = model.features(anchor_img_tensor)
                    anchor_embedding = F.normalize(anchor_features, p=2, dim=1)  # shape [1, C]

                    # Select a negative sample from the current batch: any sample with a different label.
                    negative_indices = [j for j, other_label in enumerate(current_batch_labels) if other_label != positive_label]
                    if len(negative_indices) == 0:
                        continue
                    neg_idx = np.random.choice(negative_indices)
                    negative_embedding = embeddings[neg_idx].unsqueeze(0)  # shape [1, C]

                    # Compute triplet loss for this triplet: (anchor, positive, negative)
                    loss = triplet_loss_fn(anchor_embedding, positive_embedding, negative_embedding)
                    batch_triplet_loss += loss
                    n_triplets += 1

                    # For debugging, print details for the first triplet
                    if i == 0 and batch_idx == 0:
                        print(f"A: {anchor_label}, P: {positive_label}, N: {lab[neg_idx].item()} | Loss: {loss.item():.4f}")

                # Only update if we have valid triplets
                if n_triplets > 0:
                    batch_loss = batch_triplet_loss / n_triplets
                    batch_loss.backward()
                    optimizer.step()
                    running_loss += batch_loss.item()

                    # print(f"Batch {batch_idx + 1} Loss: {batch_loss.item():.4f}")
                else:
                    print(f"Batch {batch_idx + 1}: No valid triplets found.")

                wandb.log({
                    "loss": batch_loss.item(),
                    "batch": batch_idx + 1
                })

            epoch_loss = running_loss / (batch_idx + 1)
            print(f"Epoch {epoch + 1} Average Loss: {epoch_loss:.4f}")

            wandb.log({
                "epoch_loss": epoch_loss,
                "epoch": epoch + 1
            })

            all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, C]
            all_labels = torch.cat(all_labels, dim=0)          # [N]

            epoch_map, class_map = helpers.compute_map_per_class(all_embeddings, all_labels)

            print(f"Epoch {epoch + 1} Overall mAP: {epoch_map:.4f}")
            print("Per-Class mAP:")
            for class_id, ap in class_map.items():
                print(f"  Class {class_id}: mAP = {ap:.4f}")

            # Log to Weights & Biases
            wandb_log_dict = {
                "epoch_loss": epoch_loss,
                "epoch_map": epoch_map,
                "epoch": epoch + 1
            }
            for class_id, ap in class_map.items():
                wandb_log_dict[f"class_{class_id}_map"] = ap

            wandb.log(wandb_log_dict)

            torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join("checkpoints", f"cl-final_model.pth"))






    except KeyError as e:
        print(f"Missing configuration: {e}")