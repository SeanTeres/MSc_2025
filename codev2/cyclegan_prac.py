import sys
import os


# Add mbod-data-processor to the Python path
sys.path.append(os.path.abspath("../mbod-data-processor"))

sys.path.append(os.path.abspath("/home/sean/MSc_2025/codev2"))

from datasets.hdf_dataset import HDF5Dataset, HDF5Dataset2, HDF5DatasetCombined
from utils import LABEL_SCHEMES, load_config
from data_splits import stratify, get_label_scheme_supports
import numpy as np
import matplotlib.pyplot as plt
import h5py
from datasets.dataloader import get_dataloaders
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
import math
import random
import pandas as pd
from datetime import datetime

from cyclegan.generator import Generator  # Import the Generator class
from cyclegan.discriminator import Discriminator  # Import the Discriminator class
from cyclegan.cyclegan import CycleGAN  # Import the CycleGAN class

from datasets.utils import ILO_CLASSIFICATION_DICTIONARY
from tsne import visualize_tsne, MultiClassBaseClassifier, extract_model_type, visualize_tsne_with_kaggle_tb

from medvae.medvae_main import MVAE
from datasets.kaggle_tb import KaggleTBDataset  # Import the KaggleTBDataset class


def plot_images(images, titles=None, cols=4):
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.flatten() if rows > 1 else [axes]
    for i, img in enumerate(images):
        img_np = img.squeeze().cpu().numpy()
        axes[i].imshow(img_np, cmap='gray')
        axes[i].axis('off')
        if titles:
            axes[i].set_title(titles[i])
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig("cycle_gan_results.png")
    plt.show()

def train_cyclegan(
    cycle_gan,
    vae_model,
    train_loader_mbod,
    train_loader_tb,
    val_loader_mbod,
    val_loader_tb,
    device,
    num_epochs=10,
    lr=0.0002,
    beta1=0.5,
    lambda_cycle=10.0,
    save_interval=5,
    sample_interval=2
):
    """
    Train the CycleGAN model for domain adaptation between MBOD and TB datasets.
    
    Args:
        cycle_gan: CycleGAN model
        vae_model: VAE model for encoding/decoding images
        train_loader_mbod: DataLoader for MBOD training data
        train_loader_tb: DataLoader for TB training data
        val_loader_mbod: DataLoader for MBOD validation data
        val_loader_tb: DataLoader for TB validation data
        device: Device to use for training (CPU/GPU)
        num_epochs: Number of training epochs
        lr: Learning rate
        beta1: Beta1 parameter for Adam optimizer
        lambda_cycle: Weight for cycle consistency loss
        save_interval: Interval for saving model checkpoints
        sample_interval: Interval for generating sample images
    """
    # Create directory for saving results
    os.makedirs("cyclegan_results", exist_ok=True)
    os.makedirs("cyclegan_checkpoints", exist_ok=True)
    
    # Initialize optimizers
    optimizer_G = torch.optim.Adam(
        list(cycle_gan.gen_MBOD2TB.parameters()) + list(cycle_gan.gen_TB2MBOD.parameters()),
        lr=lr,
        betas=(beta1, 0.999)
    )
    optimizer_D = torch.optim.Adam(
        list(cycle_gan.discriminator_TB.parameters()) + list(cycle_gan.discriminator_MBOD.parameters()),
        lr=lr,
        betas=(beta1, 0.999)
    )
    
    # Learning rate scheduler
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=lambda epoch: 1.0 - max(0, epoch - num_epochs // 2) / (num_epochs // 2)
    )
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D, lr_lambda=lambda epoch: 1.0 - max(0, epoch - num_epochs // 2) / (num_epochs // 2)
    )
    
    # Loss functions
    criterion_GAN = torch.nn.MSELoss()  # For adversarial loss
    criterion_cycle = torch.nn.L1Loss()  # For cycle consistency loss
    
    # Extract encoder and decoder from VAE
    ae = vae_model.model
    ae.eval()  # Keep VAE in eval mode
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        cycle_gan.train()
        
        # Create iterators for both dataloaders
        mbod_iter = iter(train_loader_mbod)
        tb_iter = iter(train_loader_tb)
        
        # Get the number of batches in each dataloader
        num_mbod_batches = len(train_loader_mbod)
        num_tb_batches = len(train_loader_tb)
        max_batches = max(num_mbod_batches, num_tb_batches)
        
        # Track losses for reporting
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_cycle_loss = 0.0
        epoch_gan_loss_MBOD2TB = 0.0
        epoch_gan_loss_TB2MBOD = 0.0
        epoch_d_tb_loss = 0.0
        epoch_d_mbod_loss = 0.0
        
        # Track discriminator accuracy
        total_real_correct = 0
        total_fake_correct = 0
        total_samples = 0
        
        batch_count = 0
        
        # Iterate for the maximum number of batches
        for i in range(max_batches):
            batch_count += 1
            
            # Get MBOD batch (restart iterator if exhausted)
            try:
                mbod_batch = next(mbod_iter)
            except StopIteration:
                mbod_iter = iter(train_loader_mbod)
                mbod_batch = next(mbod_iter)
            
            # Get TB batch (restart iterator if exhausted)
            try:
                tb_batch = next(tb_iter)
            except StopIteration:
                tb_iter = iter(train_loader_tb)
                tb_batch = next(tb_iter)
            
            # Unpack the batches
            mbod_imgs, _ = mbod_batch
            tb_imgs, _ = tb_batch
            
            # Process both batches
            mbod_imgs = mbod_imgs.to(device)
            tb_imgs = tb_imgs.to(device)
            
            # Set real and fake labels
            real_label = torch.ones(mbod_imgs.size(0), 1, 14, 14).to(device)  # Adjust size based on your discriminator output
            fake_label = torch.zeros(mbod_imgs.size(0), 1, 14, 14).to(device)
            
            # Encode images from both domains
            with torch.no_grad():
                encoded_mbod = ae.encode(mbod_imgs).sample()
                encoded_tb = ae.encode(tb_imgs).sample()
            
            # ----------------------
            # Train Generators
            # ----------------------
            optimizer_G.zero_grad()
            
            # GAN loss for generators
            fake_tb = cycle_gan.gen_MBOD2TB(encoded_mbod)
            fake_mbod = cycle_gan.gen_TB2MBOD(encoded_tb)
            
            # Discriminator evaluations for generated images
            pred_fake_tb = cycle_gan.discriminator_TB(fake_tb)
            pred_fake_mbod = cycle_gan.discriminator_MBOD(fake_mbod)
            
            # Generator losses
            gan_loss_MBOD2TB = criterion_GAN(pred_fake_tb, real_label)  # Fool TB discriminator
            gan_loss_TB2MBOD = criterion_GAN(pred_fake_mbod, real_label)  # Fool MBOD discriminator
            gan_loss = gan_loss_MBOD2TB + gan_loss_TB2MBOD
            
            # Cycle consistency loss
            # Forward cycle: MBOD -> TB -> MBOD
            recov_mbod = cycle_gan.gen_TB2MBOD(fake_tb)
            cycle_loss_mbod = criterion_cycle(recov_mbod, encoded_mbod) * lambda_cycle
            
            # Backward cycle: TB -> MBOD -> TB
            recov_tb = cycle_gan.gen_MBOD2TB(fake_mbod)
            cycle_loss_tb = criterion_cycle(recov_tb, encoded_tb) * lambda_cycle
            
            # Total cycle consistency loss
            cycle_loss = cycle_loss_mbod + cycle_loss_tb
            
            # Total generator loss
            g_loss = gan_loss + cycle_loss
            g_loss.backward()
            optimizer_G.step()
            
            # ----------------------
            # Train Discriminators
            # ----------------------
            optimizer_D.zero_grad()
            
            # TB Discriminator
            # Real TB images
            pred_real_tb = cycle_gan.discriminator_TB(encoded_tb)
            loss_d_real_tb = criterion_GAN(pred_real_tb, real_label)
            
            # Fake TB images (previously generated)
            pred_fake_tb = cycle_gan.discriminator_TB(fake_tb.detach())
            loss_d_fake_tb = criterion_GAN(pred_fake_tb, fake_label)
            
            # Combined TB discriminator loss
            loss_d_tb = (loss_d_real_tb + loss_d_fake_tb) * 0.5
            
            # MBOD Discriminator
            # Real MBOD images
            pred_real_mbod = cycle_gan.discriminator_MBOD(encoded_mbod)
            loss_d_real_mbod = criterion_GAN(pred_real_mbod, real_label)
            
            # Fake MBOD images (previously generated)
            pred_fake_mbod = cycle_gan.discriminator_MBOD(fake_mbod.detach())
            loss_d_fake_mbod = criterion_GAN(pred_fake_mbod, fake_label)
             
            # Combined MBOD discriminator loss
            loss_d_mbod = (loss_d_real_mbod + loss_d_fake_mbod) * 0.5
            
            # Total discriminator loss
            d_loss = loss_d_tb + loss_d_mbod
            d_loss.backward()
            optimizer_D.step()
            
            # Calculate discriminator accuracy
            with torch.no_grad():
                # Count correct predictions for real images
                real_correct = (pred_real_tb > 0.5).float().sum() + (pred_real_mbod > 0.5).float().sum()
                # Count correct predictions for fake images
                fake_correct = (pred_fake_tb < 0.5).float().sum() + (pred_fake_mbod < 0.5).float().sum()
                # Total number of predictions
                num_samples = pred_real_tb.numel() + pred_real_mbod.numel() + pred_fake_tb.numel() + pred_fake_mbod.numel()
                
                total_real_correct += real_correct.item()
                total_fake_correct += fake_correct.item()
                total_samples += num_samples
            
            # Update running loss totals
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_cycle_loss += cycle_loss.item()
            epoch_gan_loss_MBOD2TB += gan_loss_MBOD2TB.item()
            epoch_gan_loss_TB2MBOD += gan_loss_TB2MBOD.item()
            epoch_d_tb_loss += loss_d_tb.item()
            epoch_d_mbod_loss += loss_d_mbod.item()
            
            # Print progress
            if (i + 1) % 50 == 0:
                print(f"Batch {i+1}/{max_batches}: G_loss: {g_loss.item():.4f}, D_loss: {d_loss.item():.4f}")
                print(f"  G details: GAN={gan_loss.item():.4f}, Cycle={cycle_loss.item():.4f}")
                print(f"  D details: D_TB={loss_d_tb.item():.4f}, D_MBOD={loss_d_mbod.item():.4f}")
                
                wandb.log({
                    "batch": i+1,  # Add global batch counter
                    "batch_G_MBOD2TB_loss": gan_loss_MBOD2TB.item(),
                    "batch_G_TB2MBOD_loss": gan_loss_TB2MBOD.item(),
                    "batch_G_cycle_loss": cycle_loss.item(),
                    "batch_D_TB_loss": loss_d_tb.item(),
                    "batch_D_MBOD_loss": loss_d_mbod.item(),
                    "batch_G_loss": g_loss.item(),
                    "batch_D_loss": d_loss.item(),
                })
                # Clear GPU memory
                torch.cuda.empty_cache()
        
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D.step()
        
        # Calculate epoch averages
        avg_g_loss = epoch_g_loss / batch_count
        avg_d_loss = epoch_d_loss / batch_count
        avg_cycle_loss = epoch_cycle_loss / batch_count
        avg_gan_loss_MBOD2TB = epoch_gan_loss_MBOD2TB / batch_count
        avg_gan_loss_TB2MBOD = epoch_gan_loss_TB2MBOD / batch_count
        avg_d_tb_loss = epoch_d_tb_loss / batch_count
        avg_d_mbod_loss = epoch_d_mbod_loss / batch_count
        
        # Calculate discriminator accuracy
        disc_acc_real = total_real_correct / (total_samples/2) * 100
        disc_acc_fake = total_fake_correct / (total_samples/2) * 100
        disc_acc_total = (total_real_correct + total_fake_correct) / total_samples * 100
        
        # Log epoch-level metrics
        wandb.log({
            "epoch": epoch + 1,
            "epoch_G_loss": avg_g_loss,
            "epoch_D_loss": avg_d_loss,
            "epoch_cycle_loss": avg_cycle_loss,
            "epoch_G_MBOD2TB_loss": avg_gan_loss_MBOD2TB,
            "epoch_G_TB2MBOD_loss": avg_gan_loss_TB2MBOD,
            "epoch_D_TB_loss": avg_d_tb_loss,
            "epoch_D_MBOD_loss": avg_d_mbod_loss,
            "disc_acc_real": disc_acc_real,
            "disc_acc_fake": disc_acc_fake,
            "disc_acc_total": disc_acc_total,
            "learning_rate": optimizer_G.param_groups[0]['lr']
        })
        
        # Print epoch results
        avg_g_loss = epoch_g_loss / batch_count
        avg_d_loss = epoch_d_loss / batch_count
        print(f"Epoch {epoch+1} - Avg G_loss: {avg_g_loss:.4f}, Avg D_loss: {avg_d_loss:.4f}")
        print(f"Discriminator accuracy - Real: {disc_acc_real:.2f}%, Fake: {disc_acc_fake:.2f}%, Total: {disc_acc_total:.2f}%")
        
        # Save sample images
        if (epoch + 1) % sample_interval == 0:
            cycle_gan.eval()
            with torch.no_grad():
                # Get sample images from validation set
                mbod_sample_batch = next(iter(val_loader_mbod))
                tb_sample_batch = next(iter(val_loader_tb))
                
                mbod_sample = mbod_sample_batch[0].to(device)[:4]  # Take first 4 samples
                tb_sample = tb_sample_batch[0].to(device)[:4]  # Take first 4 samples
                
                # Encode images
                encoded_mbod = ae.encode(mbod_sample).sample()
                encoded_tb = ae.encode(tb_sample).sample()
                
                # Generate translations
                fake_tb = cycle_gan.gen_MBOD2TB(encoded_mbod)
                fake_mbod = cycle_gan.gen_TB2MBOD(encoded_tb)
                
                # Generate reconstructions
                recov_mbod = cycle_gan.gen_TB2MBOD(fake_tb)
                recov_tb = cycle_gan.gen_MBOD2TB(fake_mbod)
                
                # Decode to get full resolution images
                fake_tb_img = ae.decode(fake_tb)
                fake_mbod_img = ae.decode(fake_mbod)
                recov_mbod_img = ae.decode(recov_mbod)
                recov_tb_img = ae.decode(recov_tb)
                
                # Prepare images for visualization
                images = []
                for i in range(4):  # For each of the 4 samples
                    images.extend([
                        mbod_sample[i].cpu(),
                        tb_sample[i].cpu(),
                        fake_tb_img[i].cpu(),
                        fake_mbod_img[i].cpu(),
                        recov_mbod_img[i].cpu(),
                        recov_tb_img[i].cpu()
                    ])
                
                # Create image grid
                n_row = 6  # 6 types of images
                n_col = 4  # 4 samples
                fig, axes = plt.subplots(n_row, n_col, figsize=(12, 18))
                
                titles = ["MBOD Original", "TB Original", 
                          "MBOD→TB", "TB→MBOD", 
                          "MBOD→TB→MBOD", "TB→MBOD→TB"]
                
                for i, (img, ax) in enumerate(zip(images, axes.flatten())):
                    ax.imshow(img.squeeze().numpy(), cmap='gray')
                    ax.axis('off')
                    if i < 6:  # Only add titles to the first column
                        ax.set_title(titles[i])
                
                plt.tight_layout()
                plt.savefig(f"cyclegan_results/samples_epoch_{epoch+1}.png")
                plt.close()
                
        # Save model checkpoints
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs:
            torch.save({
                'generator_MBOD2TB': cycle_gan.gen_MBOD2TB.state_dict(),
                'generator_TB2MBOD': cycle_gan.gen_TB2MBOD.state_dict(),
                'discriminator_TB': cycle_gan.discriminator_TB.state_dict(),
                'discriminator_MBOD': cycle_gan.discriminator_MBOD.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'epoch': epoch
            }, f"cyclegan_checkpoints/cyclegan_checkpoint_epoch_{epoch+1}.pth")
            
    print("Training complete!")
    return cycle_gan


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("*" * 50)
    print(f"Using device: {device}")
    print("*" * 50)
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    config = load_config("/home/sean/MSc_2025/codev2/config.yaml")
    
    preprocess = transforms.Compose([
    # transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
    # transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    try:
        # Get the path to the generated HDF5 file
        hdf5_file_path = config["merged_silicosis_output"]["hdf5_file"]
        ilo_hdf5_file_path = config["ilo_output"]["hdf5_file"]

                # Path to Kaggle TB dataset
        kaggle_tb_path = config["kaggle_TB"]["outputpath"]  # Ensure this is set in config.yaml

        # Create an instance of KaggleTBDataset
        kaggle_tb_dataset = HDF5Dataset(
            hdf5_path = kaggle_tb_path,
            labels_key="tuberculosis",
            preprocess = preprocess
        )
     

        # Create an HDF5SilicosisDataset instance
        mbod_dataset = HDF5DatasetCombined(
            hdf5_path=hdf5_file_path,
            labels_key="multiclass_stb",  # Main pathology labels, 'lab' for all labels
            images_key="images",
            augmentations=None,
            preprocess=preprocess
        )
        ilo_dataset = HDF5Dataset2(
            hdf5_path=ilo_hdf5_file_path,
            labels_key="profusion_score",  # Main pathology labels, 'lab' for all labels
            images_key="images",
            augmentations=None,
            preprocess=preprocess
        )


        wandb.login()
        wandb.init(project='MBOD-cyclegan', name='init_test',
                   config={
                       "experiment_type:": "CycleGAN - Base",
                       "batch_size": 4,
                       "n_epochs": 100,
                       "learning_rate": 0.0002,
                       "beta1": 0.5,
                       "lambda_cycle": 10.0,
                   })



        vae_model = MVAE(
        model_name='medvae_4_1_2d',
        modality='xray',
        ).to(device)
        vae_model.requires_grad_(False)
        vae_model.eval()

        ae = vae_model.model

        # Getting the transform and applying it
        transform = vae_model.get_transform()




        # Retrieve the labels

        # Define the mapping for multiclass_stb
        multiclass_stb_mapping = {
            0: "Profusion 0, No TB",
            1: "Profusion 1, No TB",
            2: "Profusion 2, No TB",
            3: "Profusion 3, No TB",
            4: "Profusion 0, With TB",
            5: "Profusion 1, With TB",
            6: "Profusion 2, With TB",
            7: "Profusion 3, With TB",
        }

        oversample = True
        batch_size = 16



        augmentations_list = transforms.Compose([
            transforms.RandomRotation(degrees=10, expand=False, fill=0),
            # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0)
        ])

        # Get the dataloaders
        train_loader, _, _ = get_dataloaders(
            hdf5_path=hdf5_file_path,
            preprocess=preprocess,
            batch_size=4,
            labels_key="multiclass_stb",
            split_file="stratified_split_mstb_new.json",
            augmentations=augmentations_list,
            oversample=None,
            balanced_batches=False
        )

        _, val_loader, test_loader = get_dataloaders(
            hdf5_path=hdf5_file_path,
            preprocess=preprocess,
            batch_size=4,
            labels_key="multiclass_stb",
            split_file="stratified_split_mstb_new.json",
            augmentations=None,
            oversample=None,
            balanced_batches=False
        )

        train_loader_tb, _, _ = get_dataloaders(
            hdf5_path=kaggle_tb_path,
            preprocess=preprocess,
            batch_size=4,
            labels_key="tuberculosis",
            split_file="stratified_split_tb_net.json",
            augmentations=None,
            oversample=None,
            balanced_batches=False
        )

        _, val_loader_tb, test_loader_tb = get_dataloaders(
            hdf5_path=kaggle_tb_path,
            preprocess=preprocess,
            batch_size=4,
            labels_key="tuberculosis",
            split_file="stratified_split_tb_net.json",
            augmentations=None,
            oversample=None,
            balanced_batches=False
        )
        
        
        # Initialize model architecture (same as used during training)
        model = xrv.models.ResNet(weights="resnet50-res512-all")
        # # Print label distributions 
        # print("\n===== TRAIN DATALOADER =====")
        # print_dataloader_label_distribution(train_loader, multiclass_stb_mapping)

        # print("\n===== VALIDATION DATALOADER =====")
        # print_dataloader_label_distribution(val_loader, multiclass_stb_mapping) 

        # print("\n===== TEST DATALOADER =====")
        # print_dataloader_label_distribution(test_loader, multiclass_stb_mapping)

        print(len(mbod_dataset))

        x = mbod_dataset[0]
        y = kaggle_tb_dataset[200]

        print(f"\n {x[1]} \n")
        print(f"\n {y[1]} \n")

        img = x[0].to(device)
        if len(img.shape) == 3:  # [C, H, W]
            img = img.unsqueeze(0)  # [1, C, H, W]

        
        # 1. Encode images from both domains
        with torch.no_grad():
            encoded_tb_net = ae.encode(y[0].unsqueeze(0).to(device)).sample()  # [batch, 1, 128, 128]
            encoded_mbod = ae.encode(x[0].unsqueeze(0).to(device)).sample()  # [batch, 1, 128, 128]

        cycle_gan = CycleGAN(img_channels=1).to(device)  # Initialize CycleGAN with 1 channel for grayscale images

       

        # 2. Apply CycleGAN generators to transform between domains
        # Use your existing Generator class
        cycle_gan.gen_TB2MBOD = Generator(img_channels=1).to(device)
        cycle_gan.gen_MBOD2TB = Generator(img_channels=1).to(device)

        # Initialize discriminators for both domains
        cycle_gan.discriminator_TB = Discriminator(in_channels=1).to(device)  # For TB-Net domain
        cycle_gan.discriminator_MBOD = Discriminator(in_channels=1).to(device)  # For MBOD domain


        fake_encoded_mbod = cycle_gan.gen_TB2MBOD(encoded_tb_net)  # A->B in encoded space
        fake_encoded_tb_net = cycle_gan.gen_MBOD2TB(encoded_mbod)  # B->A in encoded space

        # 3. Apply cycle consistency
        reconstructed_encoded_tb_net = cycle_gan.gen_MBOD2TB(fake_encoded_mbod)  # A->B->A
        reconstructed_encoded_mbod = cycle_gan.gen_TB2MBOD(fake_encoded_tb_net)  # B->A->B

        # 4. Decode to get full resolution images
        with torch.no_grad():
            fake_images_mbod = ae.decode(fake_encoded_mbod)
            fake_images_tb_net = ae.decode(fake_encoded_tb_net)
            reconstructed_images_tb_net = ae.decode(reconstructed_encoded_tb_net)
            reconstructed_images_mbod = ae.decode(reconstructed_encoded_mbod)
        

        
        # Prepare images and titles for visualization
        images = [
            x[0].cpu(),  # Original image from domain B
            y[0].cpu(),  # Original image from domain A
            fake_images_mbod.squeeze().cpu(),  # Fake image in domain B
            fake_images_tb_net.squeeze().cpu(),  # Fake image in domain A
            reconstructed_images_tb_net.squeeze().cpu(),  # Reconstructed image in domain A
            reconstructed_images_mbod.squeeze().cpu()   # Reconstructed image in domain B
        ]
        titles = [
            "Original Image MBOD",
            "Original Image TB-Net",
            "Fake Image in Domain MBOD",
            "Fake Image in Domain TB-Net",
            "Reconstructed Image in Domain TB-Net",
            "Reconstructed Image in Domain MBOD"
        ]

       #  plot_images(images, titles=titles, cols=2)


        # Test the discriminators on encoded images
        with torch.no_grad():
            # Discriminate real images
            disc_real_TB = cycle_gan.discriminator_TB(encoded_tb_net)
            disc_real_MBOD = cycle_gan.discriminator_MBOD(encoded_mbod)

            # Discriminate fake images
            disc_fake_TB = cycle_gan.discriminator_TB(fake_encoded_tb_net)
            disc_fake_MBOD = cycle_gan.discriminator_MBOD(fake_encoded_mbod)

            # Print discriminator outputs
            print(f"Discriminator A (TB-Net domain) - Real: {disc_real_TB.mean().item():.4f}, Fake: {disc_fake_TB.mean().item():.4f}")
            print(f"Discriminator B (MBOD domain) - Real: {disc_real_MBOD.mean().item():.4f}, Fake: {disc_fake_MBOD.mean().item():.4f}")



        # Create iterators for both dataloaders
        # mbod_iter = iter(train_loader)
        # tb_iter = iter(train_loader_tb)

        # Get the number of batches in each dataloader
        # num_mbod_batches = len(train_loader)
        # num_tb_batches = len(train_loader_tb)
        # max_batches = max(num_mbod_batches, num_tb_batches)

        # print(f"MBOD batches: {num_mbod_batches}, TB batches: {num_tb_batches}")


        # Iterate for the maximum number of batches
        # for i in range(max_batches):
        #     # Get MBOD batch (restart iterator if exhausted)
        #     try:
        #         mbod_batch = next(mbod_iter)
        #     except StopIteration:
        #         mbod_iter = iter(train_loader)
        #         mbod_batch = next(mbod_iter)
            
        #     # Get TB batch (restart iterator if exhausted)
        #     try:
        #         tb_batch = next(tb_iter)
        #     except StopIteration:
        #         tb_iter = iter(train_loader_tb)
        #         tb_batch = next(tb_iter)
            
        #     # Unpack the batches
        #     mbod_imgs, mbod_labels = mbod_batch
        #     tb_imgs = tb_batch[0]  # Assuming tb_batch is a tuple (images, labels) or just images
        #     tb_labels = tb_batch[1] if len(tb_batch) > 1 else None  # Check if labels are present
            
        #     # Process both batches
        #     mbod_imgs = mbod_imgs.to(device)
        #     tb_imgs = tb_imgs.to(device)      

        #     # Encode images from both domains
        #     with torch.no_grad():
        #         encoded_mbod = ae.encode(mbod_imgs).sample()
        #         encoded_tb_net = ae.encode(tb_imgs).sample()

        #     # Apply CycleGAN generators to transform between domains
        #     fake_encoded_mbod = cycle_gan.gen_TB2MBOD(encoded_tb_net)
        #     fake_encoded_tb_net = cycle_gan.gen_MBOD2TB(encoded_mbod)

        #     # Decode to get full resolution images
        #     with torch.no_grad():
        #         fake_images_mbod = ae.decode(fake_encoded_mbod)
        #         fake_images_tb_net = ae.decode(fake_encoded_tb_net)
        #         print(fake_images_mbod.shape, fake_images_tb_net.shape)


        #     # Visualize results for this batch
        #     plot_images(
        #         [
        #             mbod_imgs[0].cpu(),  # Original MBOD image
        #             tb_imgs[0].cpu(),    # Original TB-Net image
        #             fake_images_mbod[0].cpu(),  # Fake MBOD image
        #             fake_images_tb_net[0].cpu()  # Fake TB-Net image
        #         ],
        #         titles=[
        #             "Original MBOD Image",
        #             "Original TB-Net Image",
        #             "Fake MBOD Image",
        #             "Fake TB-Net Image"
        #         ],
        #         cols=2
        #     )


        # Initialize learning parameters
        lr = wandb.config.learning_rate  # Set learning rate from wandb config
        beta1 = wandb.config.beta1
        lambda_cycle = wandb.config.lambda_cycle
        
        try:
            print("Starting CycleGAN training...")
            trained_model = train_cyclegan(
                cycle_gan=cycle_gan,
                vae_model=vae_model,
                train_loader_mbod=train_loader,
                train_loader_tb=train_loader_tb,
                val_loader_mbod=val_loader,
                val_loader_tb=val_loader_tb,
                device=device,
                num_epochs=wandb.config.n_epochs,
                lr=lr,
                beta1=beta1,
                lambda_cycle=lambda_cycle,
                save_interval=5,
                sample_interval=2
            )
            
            print("CycleGAN training completed!")
            
            # Save the final model
            torch.save({
                'generator_MBOD2TB': trained_model.gen_MBOD2TB.state_dict(),
                'generator_TB2MBOD': trained_model.gen_TB2MBOD.state_dict(),
                'discriminator_TB': trained_model.discriminator_TB.state_dict(),
                'discriminator_MBOD': trained_model.discriminator_MBOD.state_dict()
            }, "cyclegan_checkpoints/cyclegan_final_model.pth")
            
            print("Final model saved!")

        except KeyboardInterrupt:
            print("Training interrupted by user. Saving current model...")
            torch.save({
                'generator_MBOD2TB': cycle_gan.gen_MBOD2TB.state_dict(),
                'generator_TB2MBOD': cycle_gan.gen_TB2MBOD.state_dict(),
                'discriminator_TB': cycle_gan.discriminator_TB.state_dict(),
                'discriminator_MBOD': cycle_gan.discriminator_MBOD.state_dict()
            }, "cyclegan_checkpoints/cyclegan_interrupted.pth")
            
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()




    except KeyError as e:
        print(f"Missing configuration: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
