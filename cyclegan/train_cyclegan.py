import itertools

import h5py
from scipy import ndimage
import wandb
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

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torch.nn.functional as F

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

def log_model_stats(model):
    # Gradient norms
    g_grad_norm = 0.0
    for p in itertools.chain(model.G_A.parameters(), model.G_B.parameters()):
        if p.grad is not None:
            g_grad_norm += p.grad.norm().item() ** 2
    g_grad_norm = g_grad_norm ** 0.5
    
    d_a_grad_norm = sum(p.grad.norm().item() ** 2 for p in model.D_A.parameters() if p.grad is not None) ** 0.5
    d_b_grad_norm = sum(p.grad.norm().item() ** 2 for p in model.D_B.parameters() if p.grad is not None) ** 0.5
    
    return {
        'model/generator_grad_norm': g_grad_norm,
        'model/disc_A_grad_norm': d_a_grad_norm,
        'model/disc_B_grad_norm': d_b_grad_norm,
    }

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
                    lambda_A = 10.0
                    lambda_B = 10.0
                    batch_G_loss = GAN_loss + Cycle_loss + 0.5 * identity_loss
                    total_G_loss += batch_G_loss.item()
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
                            f"{name}-_batch{idx}", 
                            sample_A, sample_B, 
                            sample_fake_A, sample_fake_B,
                            sample_rec_A, sample_rec_B,
                            log_to_wandb=True
                        )



                pbar.set_description(f"Epoch {epoch} - G: {total_G_loss.item():.4f}, D_A: {D_A_loss.item():.4f}, D_B: {D_B_loss.item():.4f}")

        avg_G_loss = total_G_loss / len(dataloader_A)
        avg_D_A_loss = total_D_A_loss / len(dataloader_A)
        avg_D_B_loss = total_D_B_loss / len(dataloader_A)
        print(f"Epoch [{epoch}] - Avg G Loss: {avg_G_loss:.4f}, Avg D_A Loss: {avg_D_A_loss:.4f}, Avg D_B Loss: {avg_D_B_loss:.4f}")
        # At the end of the function:

        wandb.log({
        'epoch/generator_loss': avg_G_loss,
        'epoch/disc_A_loss': avg_D_A_loss,
        'epoch/disc_B_loss': avg_D_B_loss,
    }, step=epoch)
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
    total_cycle_loss = 0.0
    total_gan_loss = 0.0
    total_identity_loss = 0.0

    pbar = tqdm(loader, total=len(loader))

    for idx, (data_A, data_B) in enumerate(pbar):
        # Unpack images 
        real_A, _ = data_A
        real_B, _ = data_B
        real_A = real_A.to(device)
        real_B = real_B.to(device)


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
            batch_G_loss = GAN_loss + Cycle_loss + 0.5 * identity_loss
            total_G_loss += batch_G_loss.item()
        # Update Generators
        scaler.scale(batch_G_loss).backward()

        if idx % 250 == 0:  # Log every 100 batches to avoid too many points
            with torch.no_grad():
                grad_stats = log_model_stats(model)
                wandb.log(grad_stats, step=epoch * len(loader) + idx)
    
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

        total_cycle_loss += Cycle_loss.item()
        total_gan_loss += GAN_loss.item()
        total_identity_loss += identity_loss.item()

        # Optional visualization every 300 batches
        if idx % 250 == 0:


            wandb.log({
                'batch/generator_loss': batch_G_loss.item(),
                'batch/disc_A_loss': D_A_loss.item(),
                'batch/disc_B_loss': D_B_loss.item(),
                'batch/cycle_loss': Cycle_loss.item(),
                'batch/identity_loss': identity_loss.item(),
                'batch/gan_loss': GAN_loss.item()
            })

            with torch.no_grad():
                sample_A = real_A[:1]
                sample_B = real_B[:1]
                sample_fake_A, sample_fake_B, sample_rec_A, sample_rec_B = model(sample_A, sample_B)
                visualize_results(
                    f"{name}-e{epoch}-b{idx}", 
                    sample_A, sample_B, 
                    sample_fake_A, sample_fake_B,
                    sample_rec_A, sample_rec_B,
                    log_to_wandb=True
                )

        pbar.set_description(f"Epoch {epoch} - G: {batch_G_loss.item():.4f}, D_A: {D_A_loss.item():.4f}, D_B: {D_B_loss.item():.4f}")

    # Compute average losses
    avg_G_loss = total_G_loss / len(loader)
    avg_D_A_loss = total_D_A_loss / len(loader)
    avg_D_B_loss = total_D_B_loss / len(loader)
    avg_cycle_loss = total_cycle_loss / len(loader)
    avg_gan_loss = total_gan_loss / len(loader) 
    avg_identity_loss = total_identity_loss / len(loader)

    grad_stats = log_model_stats(model)
    wandb.log(grad_stats, step=epoch)



    print(f"Epoch [{epoch}] - Avg G Loss: {avg_G_loss:.4f}, Avg D_A Loss: {avg_D_A_loss:.4f}, Avg D_B Loss: {avg_D_B_loss:.4f}")

    return {
        'generator_loss': avg_G_loss,
        'discriminator_A_loss': avg_D_A_loss,
        'discriminator_B_loss': avg_D_B_loss,
        'cycle_loss': avg_cycle_loss,
        'gan_loss': avg_gan_loss,
        'identity_loss': avg_identity_loss
    }


def visualize_results(epoch, real_A, real_B, fake_A, fake_B, recon_A, recon_B, log_to_wandb=True):
    """Save visualization of generated images"""
    fig, axs = plt.subplots(2, 3, figsize=(10, 10))

    fig.suptitle(f'CycleGAN Results: {epoch}', fontsize=16)
    
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
        wandb.log({f"CycleGAN Results": wandb.Image(fig)})
        print(f"Logged CycleGAN results for epoch {epoch} to Weights & Biases.")
    
    plt.tight_layout()
    # plt.savefig(f'visualizations/epoch_{epoch}.png')
    plt.close()



def evaluate_I2I_metrics(model, val_loader, device, epoch, max_batches=15, set_name="val"):
    """
    Pure evaluation: returns dict; caller handles logging.
    """
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure
    model.eval()
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    ssim_A2B_values=[]; ssim_B2A_values=[]
    psnr_A2B_values=[]; psnr_B2A_values=[]
    ms_ssim_A2B_values=[]; ms_ssim_B2A_values=[]
    mae_A2B_values=[]; mae_B2A_values=[]

    with torch.no_grad():
        for batch_idx, (data_A, data_B) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            real_A,_ = data_A
            real_B,_ = data_B
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            fake_A = model.G_B(real_B)
            fake_B = model.G_A(real_A)

            real_A_norm = (real_A + 1)/2.0
            real_B_norm = (real_B + 1)/2.0
            fake_A_norm = (fake_A + 1)/2.0
            fake_B_norm = (fake_B + 1)/2.0

            ssim_A2B_values.append(ssim(fake_B_norm, real_A_norm).item())
            ssim_B2A_values.append(ssim(fake_A_norm, real_B_norm).item())
            psnr_A2B_values.append(psnr(fake_B_norm, real_A_norm).item())
            psnr_B2A_values.append(psnr(fake_A_norm, real_B_norm).item())
            ms_ssim_A2B_values.append(ms_ssim(fake_B_norm, real_A_norm).item())
            ms_ssim_B2A_values.append(ms_ssim(fake_A_norm, real_B_norm).item())
            mae_A2B_values.append(F.l1_loss(fake_B_norm, real_A_norm).item())
            mae_B2A_values.append(F.l1_loss(fake_A_norm, real_B_norm).item())

    def m(v): return float(np.mean(v)) if v else 0.0
    metrics = {
        f"{set_name}/ssim_A2B": m(ssim_A2B_values),
        f"{set_name}/ssim_B2A": m(ssim_B2A_values),
        f"{set_name}/ssim_avg": (m(ssim_A2B_values)+m(ssim_B2A_values))/2.0,
        f"{set_name}/psnr_A2B": m(psnr_A2B_values),
        f"{set_name}/psnr_B2A": m(psnr_B2A_values),
        f"{set_name}/psnr_avg": (m(psnr_A2B_values)+m(psnr_B2A_values))/2.0,
        f"{set_name}/ms_ssim_A2B": m(ms_ssim_A2B_values),
        f"{set_name}/ms_ssim_B2A": m(ms_ssim_B2A_values),
        f"{set_name}/ms_ssim_avg": (m(ms_ssim_A2B_values)+m(ms_ssim_B2A_values))/2.0,
        f"{set_name}/mae_A2B": m(mae_A2B_values),
        f"{set_name}/mae_B2A": m(mae_B2A_values),
        f"{set_name}/mae_avg": (m(mae_A2B_values)+m(mae_B2A_values))/2.0,
    }
    for k,v in metrics.items():
        if not np.isfinite(v):
            metrics[k] = 0.0
    return metrics



def create_train_val_split(dataset, val_ratio=0.2, seed=42):
    """
    Split a dataset into train and validation subsets.
    
    Args:
        dataset: The dataset to split
        val_ratio: Fraction of data to use for validation
        seed: Random seed for reproducible splits
    
    Returns:
        train_subset, val_subset
    """
    torch.manual_seed(seed)
    dataset_size = len(dataset)
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - val_size
    
    # Create random indices
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    
    return train_subset, val_subset

if __name__ == "__main__":

    config = load_config("/home/sean/MSc_2025/cyclegan/cyclegan_cfg.yaml")

    mbod_dataset_path = config["DATA"]["DATA_PATH_MBOD"]
    rand_v3_dataset_path = config["DATA"]["DATA_PATH_RAND_V3"]
    kaggle_dataset_path = config["DATA"]["DATA_PATH_TBNET"]

    mbod_stats = stats.compute_dataset_stats(mbod_dataset_path)
    kaggle_stats = stats.compute_dataset_stats(kaggle_dataset_path)
    # rand_stats = stats.compute_dataset_stats(rand_v3_dataset_path)

    wandb.login()
    wandb.init(project=config["WANDB"]["PROJECT_NAME"], name=config["EXPERIMENT"]["NAME"], config=config)

    stats.plot_normalization_comparison(mbod_dataset_path, kaggle_dataset_path, 
                                  mbod_stats, kaggle_stats, dataset_names="MBOD_TBNET", log_to_wandb=True)
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    GAN_criterion = GANLoss('vanilla').to(device)
    Cycle_criterion = nn.L1Loss().to(device)
    Identity_criterion = nn.L1Loss().to(device)


    model = CycleGANModel(1, 1, config["GENERATOR"]["NGF"], config["DISCRIMINATOR"]["NDF"], config["GENERATOR"]["NAME"], 
                          config["DISCRIMINATOR"]["NAME"], 3, config["ARCH"]["NORM"], config["ARCH"]["INIT_TYPE"], 
                          config["ARCH"]["INIT_GAIN"], device)
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


    if config["PREPROCESS"] == "normalize":
        preprocess_mbod = transforms.Compose([
            transforms.ToTensor(),            
            transforms.Lambda(stats.create_dataset_normalizer(mbod_stats)),
            # Optionally add LoG filter here if desired
        ])
        

        preprocess_kaggle = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(stats.create_dataset_normalizer(kaggle_stats)),
            # Optionally add LoG filter here if desired
        ])
    elif config["PREPROCESS"] == "none":
        preprocess_mbod = transforms.Compose([
            transforms.ToTensor(),
        ])
        

        preprocess_kaggle = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        raise NotImplementedError("Only 'normalize' preprocessing is tested.")
    
    if config["DATALOADERS"]["TYPE"] == "unaligned":
            # Create datasets
        mbod_merged = HDF5Dataset(
            hdf5_path=mbod_dataset_path,
            labels_key="tuberculosis",
            images_key="images",
            preprocess=preprocess_mbod
        )


        # rand_v3 = HDF5Dataset(
        #     hdf5_path=rand_v3_dataset_path,
        #     labels_key="tuberculosis",
        #     images_key="images",
        #     preprocess=preprocess
        # )

        kaggle_tb = HDF5Dataset(
            hdf5_path=config["DATA"]["DATA_PATH_TBNET"],  
            labels_key="tuberculosis",
            images_key="images",
            preprocess=preprocess_kaggle
        )

        mbod_train, mbod_val = create_train_val_split(mbod_merged, val_ratio=0.2, seed=42)
        kaggle_train, kaggle_val = create_train_val_split(kaggle_tb, val_ratio=0.2, seed=42)

        print(f"Dataset splits:")
        print(f"  MBOD - Train: {len(mbod_train)}, Val: {len(mbod_val)}")
        print(f"  Kaggle - Train: {len(kaggle_train)}, Val: {len(kaggle_val)}")

        # Create training and validation datasets
        train_unaligned_dataset = UnalignedWrapper(kaggle_train, mbod_train, serial_batches=False)
        val_unaligned_dataset = UnalignedWrapper(kaggle_val, mbod_val, serial_batches=True)  # serial_batches=True for consistent validation

        # Create dataloaders
        unaligned_loader = torch.utils.data.DataLoader(
            train_unaligned_dataset, 
            batch_size=config["DATALOADERS"]["UNALIGNED_BATCH_SIZE"], 
            shuffle=True,
            num_workers=2
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_unaligned_dataset, 
            batch_size=4,  # Smaller batch size for validation
            shuffle=False,
            num_workers=2
        )

        print(f"Dataloaders:")
        print(f"  Training: {len(unaligned_loader)} batches")
        print(f"  Validation: {len(val_loader)} batches")




        for i in range(config["TRAINING"]["EPOCHS"]):


            metrics = train_one_unaligned_epoch(config["EXPERIMENT"]["NAME"], model, unaligned_loader,
                                    optimizers={'G': model.optimizer_G, 'D_A': model.optimizer_D_A, 'D_B': model.optimizer_D_B},
                                    criterions={'GAN': GAN_criterion, 'Cycle': Cycle_criterion, 'Identity': Identity_criterion},
                                    device=device, epoch=i, scaler=scaler, pretrain=False)
            

            print(f"LOGGING METRICS FOR EPOCH {i}: G={float(metrics['generator_loss']):.4f}, D_A={float(metrics['discriminator_A_loss']):.4f}, D_B={float(metrics['discriminator_B_loss']):.4f}")

            # Explicitly convert to Python floats and use a dictionary
            metrics_dict = {
                "epoch/generator_loss": float(metrics['generator_loss']),
                "epoch/disc_A_loss": float(metrics['discriminator_A_loss']),
                "epoch/disc_B_loss": float(metrics['discriminator_B_loss']),
            }
            metrics_dict.update({
                "epoch/cycle_loss": float(metrics['cycle_loss']),
                "epoch/gan_loss": float(metrics['gan_loss']),
                "epoch/identity_loss": float(metrics['identity_loss']),
            })

            metrics_dict["epoch"] = i

            global_step = (i + 1) * len(unaligned_loader)

            if i % 5 == 0:
                print(f"Running evaluation metrics at epoch {i}...")

                if i % 5 == 0:
                    print(f"Running evaluation metrics at epoch {i}...")
                    eval_metrics = evaluate_I2I_metrics(model, val_loader, device, i, max_batches=15, set_name="val")
                    wandb.log(eval_metrics, step=global_step)

            # Log all metrics at once and force a sync
            wandb.log(metrics_dict)
            wandb.run.log({})  # Force synchronization
            

    else:

    
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

        raise NotImplementedError("Paired dataloader training not fully implemented yet.")


    


        
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
# 2) Evaluate model function
# 3) Checkpointing models
# 4) Cycle & CLF metrics
# 5) Running for multiple configs
