import itertools

import h5py
from scipy import ndimage
from networks import init_net, define_D, define_G, ResnetGenerator, UnetGenerator, NLayerDiscriminator, GANLoss, init_weights
from helpers import init_weights
import stats 

from cyclegan_model import CycleGANModel
import torch
import torch.nn as nn

import sys
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
sys.path.append(os.path.abspath("../mbod-data-processor"))
from datasets.hdf_dataset import HDF5Dataset, HDF5Dataset2
from datasets.dataloader import get_dataloaders
from utils import load_config
import torchxrayvision as xrv

from tqdm import tqdm
import random
from torch.utils.data import Dataset

class UnalignedWrapper(Dataset):
    """
    Wrap two datasets (Domain A, Domain B) for unpaired I2I translation.
    Mimics CycleGAN's UnalignedDataset:
      - Returns one sample from A and one sample from B.
      - If serial_batches=True -> deterministic pairing (wrap-around).
      - If serial_batches=False -> random pairing for B.
    """
    def __init__(self, dataset_A, dataset_B, serial_batches=False):
        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        self.serial_batches = serial_batches
        self.A_size = len(dataset_A)
        self.B_size = len(dataset_B)

    def __getitem__(self, index):
        # Always cycle through A
        data_A = self.dataset_A[index % self.A_size]

        if self.serial_batches:
            # Deterministic wrap-around for B
            index_B = index % self.B_size
        else:
            # Random index for B
            index_B = random.randint(0, self.B_size - 1)

        data_B = self.dataset_B[index_B]

        return data_A, data_B

    def __len__(self):
        # Ensure the longer dataset defines the epoch size
        return max(self.A_size, self.B_size)


def train_one_epoch(name, model, dataloader_A, dataloader_B, optimizers, criterions, device, epoch, scaler=None, pretrain=False):
    model.train()

    total_G_loss = 0.0
    total_D_A_loss = 0.0
    total_D_B_loss = 0.0

    data_iterator = zip(dataloader_A, dataloader_B)
    pbar = tqdm(data_iterator, total=min(len(dataloader_A), len(dataloader_B)))

    if pretrain:
        print("Pretraining mode: Only training using reconstruction loss.")

        for i in range(2):
            for idx, (data_A, data_B) in enumerate(pbar):
                real_A = data_A[0].to(device)
                real_B = data_B[0].to(device)
                
                with torch.amp.autocast(device_type='cuda'):
                    fake_B = model.G_A(real_A)
                    rec_A = model.G_B(fake_B)
                    Cycle_loss = criterions['Cycle'](rec_A, real_A)

                    fake_A = model.G_B(real_B)
                    rec_B = model.G_A(fake_A) # added - claude
                    Cycle_loss += criterions['Cycle'](rec_B, real_B)
                
                # Only update generators with cycle loss
                scaler.scale(Cycle_loss).backward()
                scaler.step(optimizers['G'])
                scaler.update()
    else:

        for idx, (data_A, data_B) in enumerate(pbar):
            real_A = data_A[0].to(device)
            real_B = data_B[0].to(device)

            if scaler is not None:
                # Generator update with mixed precision
                with torch.amp.autocast(device_type='cuda'):
                    identity_A = model.G_A(real_B)
                    identity_B = model.G_B(real_A)
                    identity_loss = criterions['Identity'](identity_A, real_B) + criterions['Identity'](identity_B, real_A)

                    fake_A, fake_B, rec_A, rec_B = model(real_A, real_B)
                    GAN_loss = criterions['GAN'](model.D_A(fake_B), True) + criterions['GAN'](model.D_B(fake_A), True)
                    Cycle_loss = criterions['Cycle'](rec_A, real_A) + criterions['Cycle'](rec_B, real_B)
                    lambda_A = 5.0
                    lambda_B = 5.0
                    total_G_loss = GAN_loss + (lambda_A * Cycle_loss) + (0.5 * identity_loss)

                scaler.scale(total_G_loss).backward()
                scaler.step(optimizers['G'])
                scaler.update()
                optimizers['G'].zero_grad()
            
                # Update Discriminator A - ALSO NEEDS AUTOCAST
                optimizers['D_A'].zero_grad()
                with torch.amp.autocast(device_type='cuda'):  # Add this line
                    D_A_real_loss = criterions['GAN'](model.D_A(real_B), True)
                    D_A_fake_loss = criterions['GAN'](model.D_A(fake_B.detach()), False)
                    D_A_loss = (D_A_real_loss + D_A_fake_loss) * 0.5
                
                scaler.scale(D_A_loss).backward()
                scaler.step(optimizers['D_A'])
                scaler.update()
                total_D_A_loss += D_A_loss.item()

                # Update Discriminator B - ALSO NEEDS AUTOCAST
                optimizers['D_B'].zero_grad()
                with torch.amp.autocast(device_type='cuda'):  # Add this line
                    D_B_real_loss = criterions['GAN'](model.D_B(real_A), True)
                    D_B_fake_loss = criterions['GAN'](model.D_B(fake_A.detach()), False)
                    D_B_loss = (D_B_real_loss + D_B_fake_loss) * 0.5
                
                scaler.scale(D_B_loss).backward()
                scaler.step(optimizers['D_B'])
                scaler.update()
                total_D_B_loss += D_B_loss.item()

                if idx % 300 == 0:
                    with torch.no_grad():
                        # Generate samples for visualization
                        sample_A = real_A[:1]  # Take first image from batch
                        sample_B = real_B[:1]
                        sample_fake_A, sample_fake_B, sample_rec_A, sample_rec_B = model(sample_A, sample_B)
                        
                        # Visualize
                        visualize_results(
                            f"{name}-{epoch}_batch{idx}", 
                            sample_A, sample_B, 
                            sample_fake_A, sample_fake_B,
                            sample_rec_A, sample_rec_B,
                            log_to_wandb=False
                        )


                pbar.set_description(f"Epoch {epoch} - G: {total_G_loss.item():.4f}, D_A: {D_A_loss.item():.4f}, D_B: {D_B_loss.item():.4f}")

        avg_G_loss = total_G_loss / len(dataloader_A)
        avg_D_A_loss = total_D_A_loss / len(dataloader_A)
        avg_D_B_loss = total_D_B_loss / len(dataloader_A)
        print(f"Epoch [{epoch}] - Avg G Loss: {avg_G_loss:.4f}, Avg D_A Loss: {avg_D_A_loss:.4f}, Avg D_B Loss: {avg_D_B_loss:.4f}")
        # At the end of the function:
        metrics = {
            'generator_loss': avg_G_loss,
            'discriminator_A_loss': avg_D_A_loss,
            'discriminator_B_loss': avg_D_B_loss,
        }
        return metrics


def train_one_unaligned_epoch(name, model, loader, optimizers, criterions, device, epoch, scaler=None, pretrain=False):
    """
    Train one epoch of a CycleGAN-style model using a single unaligned dataloader.

    Parameters:
        model          : The CycleGAN model
        loader         : Unaligned dataloader yielding (data_A, data_B) tuples
        optimizers     : Dict with 'G', 'D_A', 'D_B'
        criterions     : Dict with 'GAN', 'Cycle', 'Identity'
        device         : Torch device
        epoch          : Current epoch number
        scaler         : GradScaler for mixed precision (optional)
        pretrain       : If True, only use reconstruction loss for pretraining
    """
    model.train()

    total_G_loss = 0.0
    total_D_A_loss = 0.0
    total_D_B_loss = 0.0

    pbar = tqdm(loader, total=len(loader))

    for idx, (data_A, data_B) in enumerate(pbar):
        # Unpack images 
        real_A, _ = data_A
        real_B, _ = data_B
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        if pretrain:
            # Only use reconstruction (cycle) loss
            with torch.amp.autocast(device_type='cuda'):
                fake_B = model.G_A(real_A)
                rec_A = model.G_B(fake_B)
                loss_A = criterions['Cycle'](rec_A, real_A)

                fake_A = model.G_B(real_B)
                rec_B = model.G_A(fake_A)
                loss_B = criterions['Cycle'](rec_B, real_B)

                total_G_loss = loss_A + loss_B

            scaler.scale(total_G_loss).backward()
            scaler.step(optimizers['G'])
            scaler.update()
            optimizers['G'].zero_grad()

        else:
            # Full CycleGAN training
            with torch.amp.autocast(device_type='cuda'):
                # Identity loss
                identity_A = model.G_A(real_B)
                identity_B = model.G_B(real_A)
                identity_loss = criterions['Identity'](identity_A, real_B) + criterions['Identity'](identity_B, real_A)

                # GAN + Cycle losses
                fake_A, fake_B, rec_A, rec_B = model(real_A, real_B)
                GAN_loss = criterions['GAN'](model.D_A(fake_B), True) + criterions['GAN'](model.D_B(fake_A), True)
                Cycle_loss = criterions['Cycle'](rec_A, real_A) + criterions['Cycle'](rec_B, real_B)
                total_G_loss = GAN_loss + Cycle_loss + 0.5 * identity_loss

            # Update Generators
            scaler.scale(total_G_loss).backward()
            scaler.step(optimizers['G'])
            scaler.update()
            optimizers['G'].zero_grad()

            # Update Discriminator A
            optimizers['D_A'].zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                D_A_real_loss = criterions['GAN'](model.D_A(real_B), True)
                D_A_fake_loss = criterions['GAN'](model.D_A(fake_B.detach()), False)
                D_A_loss = 0.5 * (D_A_real_loss + D_A_fake_loss)

                # gp_weight = 5.0
                # gradient_penalty = compute_gradient_penalty(model.D_A, real_B, fake_B.detach())
                # D_A_loss = D_A_loss + gp_weight * gradient_penalty

            scaler.scale(D_A_loss).backward()
            scaler.step(optimizers['D_A'])
            scaler.update()
            total_D_A_loss += D_A_loss.item()

            # Update Discriminator B
            optimizers['D_B'].zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                D_B_real_loss = criterions['GAN'](model.D_B(real_A), True)
                D_B_fake_loss = criterions['GAN'](model.D_B(fake_A.detach()), False)
                D_B_loss = 0.5 * (D_B_real_loss + D_B_fake_loss)
            scaler.scale(D_B_loss).backward()
            scaler.step(optimizers['D_B'])
            scaler.update()
            total_D_B_loss += D_B_loss.item()

            # Optional visualization every 300 batches
            if idx % 300 == 0:
                with torch.no_grad():
                    sample_A = real_A[:1]
                    sample_B = real_B[:1]
                    sample_fake_A, sample_fake_B, sample_rec_A, sample_rec_B = model(sample_A, sample_B)
                    visualize_results(
                        f"{name}-{epoch}-b{idx}", 
                        sample_A, sample_B, 
                        sample_fake_A, sample_fake_B,
                        sample_rec_A, sample_rec_B,
                        log_to_wandb=False
                    )

        pbar.set_description(f"Epoch {epoch} - G: {total_G_loss.item():.4f}, D_A: {D_A_loss.item():.4f}, D_B: {D_B_loss.item():.4f}")

    # Compute average losses
    avg_G_loss = total_G_loss / len(loader)
    avg_D_A_loss = total_D_A_loss / len(loader)
    avg_D_B_loss = total_D_B_loss / len(loader)

    print(f"Epoch [{epoch}] - Avg G Loss: {avg_G_loss:.4f}, Avg D_A Loss: {avg_D_A_loss:.4f}, Avg D_B Loss: {avg_D_B_loss:.4f}")

    return {
        'generator_loss': avg_G_loss,
        'discriminator_A_loss': avg_D_A_loss,
        'discriminator_B_loss': avg_D_B_loss,
    }

def visualize_results(epoch, real_A, real_B, fake_A, fake_B, recon_A, recon_B, log_to_wandb=False):
    """Save visualization of generated images"""
    fig, axs = plt.subplots(2, 3, figsize=(10, 10))
    
    axs[0, 0].imshow(real_A[0, 0].cpu().numpy(), cmap='gray')
    axs[0, 0].set_title('Real A (Intl)')
    
    axs[0, 1].imshow(fake_B[0, 0].detach().cpu().numpy(), cmap='gray')
    axs[0, 1].set_title('Fake B (A→B)')

    axs[0,2].imshow(recon_A[0, 0].detach().cpu().numpy(), cmap='gray')
    axs[0,2].set_title('Reconstructed A (A→B→A)')
    
    axs[1, 0].imshow(real_B[0, 0].cpu().numpy(), cmap='gray')
    axs[1, 0].set_title('Real B (MBOD)')
    
    axs[1, 1].imshow(fake_A[0, 0].detach().cpu().numpy(), cmap='gray')
    axs[1, 1].set_title('Fake A (B→A)')

    axs[1,2].imshow(recon_B[0, 0].detach().cpu().numpy(), cmap='gray')
    axs[1,2].set_title('Reconstructed B (B→A→B)')

    if log_to_wandb:
        import wandb
        wandb.log({f"CycleGAN Results Epoch {epoch}": wandb.Image(fig)})
        print(f"Logged CycleGAN results for epoch {epoch} to Weights & Biases.")
    
    plt.tight_layout()
    plt.savefig(f'visualizations/epoch_{epoch}.png')
    plt.close()



if __name__ == "__main__":

    config = load_config("/home/sean/MSc_2025/simclr/config.yaml")

    mbod_dataset_path = config["merged_silicosis_output"]["hdf5_file"]
    rand_v3_dataset_path = config["rand_v3_output"]["hdf5_file"]

    mbod_stats = stats.compute_dataset_stats(mbod_dataset_path)
    kaggle_stats = stats.compute_dataset_stats(config["kaggle_TB"]["outputpath"])
    # rand_stats = compute_dataset_stats(rand_v3_dataset_path)

    stats.plot_normalization_comparison(mbod_dataset_path, config["kaggle_TB"]["outputpath"], 
                                  mbod_stats, kaggle_stats, dataset_names="MBOD_TBNET", log_to_wandb=False)
    
    

    preprocess_mbod = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(stats.create_dataset_normalizer(mbod_stats))
        # Optionally add LoG filter here if desired
    ])
    
    preprocess_kaggle = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(stats.create_dataset_normalizer(kaggle_stats))
        # Optionally add LoG filter here if desired
    ])



    # Updated preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.ToTensor(),
       # transforms.Lambda(stats.create_dataset_normalizer(mbod_stats)), # Is this correct? 
       # transforms.Lambda(stats.apply_log_filter)
    ])


    # Create datasets
    mbod_merged = HDF5Dataset(
        hdf5_path=mbod_dataset_path,
        labels_key="tuberculosis",
        images_key="images",
        preprocess=preprocess_mbod
    )


    rand_v3 = HDF5Dataset(
        hdf5_path=rand_v3_dataset_path,
        labels_key="tuberculosis",
        images_key="images",
        preprocess=preprocess
    )

    kaggle_tb = HDF5Dataset(
        hdf5_path=config["kaggle_TB"]["outputpath"],  # You need this path in your config
        labels_key="tuberculosis",
        images_key="images",
        preprocess=preprocess_kaggle
    )
    
    

    
    mbod_loader_train, mbod_loader_val, mbod_loader_test = get_dataloaders(
        mbod_dataset_path,
        preprocess_mbod,
        train_split=0.8,
        batch_size=6,
        labels_key="tuberculosis",
        split_file=None,
        augmentations=None
    )
    kaggle_loader_train, kaggle_loader_val, kaggle_loader_test = get_dataloaders(
        config["kaggle_TB"]["outputpath"],
        preprocess_kaggle,
        train_split=0.8,
        batch_size=6,
        labels_key="tuberculosis",
        split_file=None,
        augmentations=None
    )

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    unaligned_dataset = UnalignedWrapper(kaggle_tb, mbod_merged, serial_batches=False) # Consider serial batches true when target domain is much larger

    # Use in a dataloader
    unaligned_loader = torch.utils.data.DataLoader(unaligned_dataset, batch_size=8, shuffle=True)




    GAN_criterion = GANLoss('vanilla').to(device)
    Cycle_criterion = nn.L1Loss().to(device)
    Identity_criterion = nn.L1Loss().to(device)


    model = CycleGANModel(1, 1, 32, 32, 'resnet_6blocks', 'basic', 3, 'batch', 'normal', 0.02, device)
    model.to(device)

    scaler = torch.GradScaler()

    # Replace the existing optimizers with:
    model.optimizer_G = torch.optim.Adam(
        itertools.chain(model.G_A.parameters(), model.G_B.parameters()), 
        lr=0.0005, betas=(0.5, 0.999)
    )
    model.optimizer_D_A = torch.optim.Adam(
        model.D_A.parameters(), 
        lr=0.0002,  # Much slower learning rate
        betas=(0.5, 0.999)
    )
    model.optimizer_D_B = torch.optim.Adam(
        model.D_B.parameters(), 
        lr=0.0002, 
        betas=(0.5, 0.999)
    )


    

    for i in range(20):

        metrics = train_one_unaligned_epoch("unaligned_gnorm", model, unaligned_loader,
                                optimizers={'G': model.optimizer_G, 'D_A': model.optimizer_D_A, 'D_B': model.optimizer_D_B},
                                criterions={'GAN': GAN_criterion, 'Cycle': Cycle_criterion, 'Identity': Identity_criterion},
                                device=device, epoch=i, scaler=scaler, pretrain=False)
        
    # for i in range(20):
    #
    #     metrics = train_one_epoch("global_norm-test",model, kaggle_loader_train, mbod_loader_train,
    #                             optimizers={'G': model.optimizer_G, 'D_A': model.optimizer_D_A, 'D_B': model.optimizer_D_B},
    #                             criterions={'GAN': GAN_criterion, 'Cycle': Cycle_criterion, 'Identity': Identity_criterion},
    #                             device=device, epoch=i, scaler=scaler)

##############################################################################
# TO DO: 
##############################################################################
# 1) Weighted Random Sampler within dataloaders to balance datasets



     