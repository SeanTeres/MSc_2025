import torch
import torch.nn as nn
from drit_model import DRIT
from torch.amp import autocast, GradScaler

def test_drit_initialization():
    """
    Basic test to ensure DRIT model initializes and runs forward pass correctly.
    """
    
    # Define a basic configuration
    cfg = {
        "INPUT_DIM_A": 1,  # Grayscale images
        "INPUT_DIM_B": 1,  # Grayscale images
        "DISC_SCALE": 1,   # Single scale discriminator (simpler)
        "DISC_NORM": "None",  # No normalization for simplicity
        "SPECTRAL_NORM": False,  # Disable spectral norm for now
        "VAE_BASED": False,  # Use simpler attribute encoder
        "LR_POLICY": "lambda",
        "EPOCHS": 100,
        "EPOCHS_DECAY": 50
    }
    torch.autograd.set_detect_anomaly(True)

    # Set device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize AMP gradient scaler
    scaler = GradScaler()
    
    # Initialize model
    print("Initializing DRIT model...")
    model = DRIT(cfg)
    model.to(device)
    
    # Initialize weights
    print("Initializing weights...")
    model.initialize_weights()
    
    # Set up schedulers
    print("Setting up schedulers...")
    model.set_scheduler(cfg)
    
    # Create smaller dummy data to avoid OOM
    batch_size = 1  # Reduced from 2
    image_A = torch.randn(batch_size, 1, 512, 512).to(device)
    image_B = torch.randn(batch_size, 1, 512, 512).to(device)
    
    print(f"Created dummy images: A={image_A.shape}, B={image_B.shape}")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Test forward pass
    print("Testing forward pass...")
    model.input_A = image_A
    model.input_B = image_B
    model.device = device
    
    try:
        # Using autocast for forward pass
        with torch.amp.autocast(device_type=device.type):
            model.forward()
        print("✅ Forward pass successful!")
        
        # Print some output shapes
        print(f"Real A encoded shape: {model.real_A_encoded.shape}")
        print(f"Fake B encoded shape: {model.fake_B_encoded.shape}")
        print(f"Content A shape: {model.z_content_A.shape}")
        print(f"Attribute A shape: {model.z_attr_A.shape}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Test translation functions
    print("Testing A→B translation...")
    try:
        with torch.amp.autocast(device_type=device.type):
            fake_B = model.test_forward(image_A[:1], A_2_B=True)
        print(f"✅ A→B translation successful! Output shape: {fake_B.shape}")
    except Exception as e:
        print(f"❌ A→B translation failed: {e}")
    
    print("Testing B→A translation...")
    try:
        with torch.amp.autocast(device_type=device.type):
            fake_A = model.test_forward(image_B[:1], A_2_B=False)
        print(f"✅ B→A translation successful! Output shape: {fake_A.shape}")
    except Exception as e:
        print(f"❌ B→A translation failed: {e}")
    
    # Test discriminator updates with AMP
    print("Testing discriminator updates...")
    try:
        torch.cuda.empty_cache()  # Clear cache before heavy operation
        
        # Modify discriminator update to use AMP
        model.input_A = image_A
        model.input_B = image_B
        
        # Forward pass with mixed precision
        with torch.amp.autocast(device_type=device.type):
            model.forward()
            
            # Update disc A
            model.disc_A_opt.zero_grad()
            loss_D1_A = model.backward_D(model.disc_A, model.real_A_encoded, model.fake_A_encoded)
            
        # Scale the loss and call backward
        scaler.scale(loss_D1_A).backward()
        scaler.step(model.disc_A_opt)
        
        # Repeat for other discriminators
        with torch.amp.autocast(device_type=device.type):
            # Update disc B
            model.disc_B_opt.zero_grad()
            loss_D1_B = model.backward_D(model.disc_B, model.real_B_encoded, model.fake_B_encoded)
            
        scaler.scale(loss_D1_B).backward()
        scaler.step(model.disc_B_opt)
        
        # Update content discriminator
        with torch.amp.autocast(device_type=device.type):
            model.disc_content_opt.zero_grad()
            loss_D_content = model.backward_D_content(model.z_content_A, model.z_content_B)
            
        scaler.scale(loss_D_content).backward()
        scaler.step(model.disc_content_opt)
        
        # Update scaler
        scaler.update()
        
        # Store loss values for reporting
        model.dis_A_loss = loss_D1_A.item()
        model.dis_B_loss = loss_D1_B.item()
        model.disc_content_loss = loss_D_content.item()
        
        print("✅ Discriminator update successful!")
        print(f"Disc A loss: {model.dis_A_loss:.4f}")
        print(f"Disc B loss: {model.dis_B_loss:.4f}")
        print(f"Disc content loss: {model.disc_content_loss:.4f}")
    except Exception as e:
        print(f"❌ Discriminator update failed: {e}")
    
    # Test generator/encoder updates with AMP
    print("Testing generator/encoder updates...")
    try:
        torch.cuda.empty_cache()  # Clear cache
        
        # Try full update with AMP
        # Forward pass
        with torch.amp.autocast(device_type=device.type):
            model.forward()
            
            # GENERATOR UPDATE
            model.enc_content_opt.zero_grad()
            model.enc_attr_opt.zero_grad()
            model.gen_opt.zero_grad()
            
            # Calculate generator losses
            loss_G_GAN_A = model.backward_G_GAN(model.fake_A_encoded, model.disc_A)
            loss_G_GAN_B = model.backward_G_GAN(model.fake_B_encoded, model.disc_B)
            
            loss_G_L1_A = model.criterionL1(model.fake_A_recon, model.real_A_encoded) * 10
            loss_G_L1_B = model.criterionL1(model.fake_B_recon, model.real_B_encoded) * 10
            
            # Total generator loss
            loss_G = loss_G_GAN_A + loss_G_GAN_B + loss_G_L1_A + loss_G_L1_B
        
        # Scale loss and backward
        scaler.scale(loss_G).backward()
        
        # Update generator weights with scaled gradients
        scaler.step(model.enc_content_opt)
        scaler.step(model.enc_attr_opt)
        scaler.step(model.gen_opt)
        
        # Update scaler
        scaler.update()
        
        print("✅ Generator/encoder update successful (with AMP)!")
        print(f"Generator GAN loss A: {loss_G_GAN_A.item():.4f}")
        print(f"Generator GAN loss B: {loss_G_GAN_B.item():.4f}")
        print(f"Reconstruction loss A: {loss_G_L1_A.item():.4f}")
        print(f"Reconstruction loss B: {loss_G_L1_B.item():.4f}")
        
    except Exception as e:
        print(f"❌ Generator/encoder update failed: {e}")
    
    print("\n🎉 Basic DRIT model test completed with AMP!")
    return True

if __name__ == "__main__":
    test_drit_initialization()