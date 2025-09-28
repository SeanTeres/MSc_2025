import torch.nn.functional as F
import torch

def nt_xent_loss(z1, z2, temperature=0.1):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)
    """
    batch_size = z1.shape[0]
    
    # Normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Concatenate z1 and z2
    z = torch.cat([z1, z2], dim=0)  # Shape: [2*batch_size, out_dim]
    
    # Compute similarity matrix
    sim_matrix = torch.mm(z, z.t()) / temperature  # Shape: [2*batch_size, 2*batch_size]
    
    # Create labels for positive pairs
    labels = torch.arange(2 * batch_size, device=z.device)
    labels = torch.cat([labels[batch_size:], labels[:batch_size]])
    
    # Mask out diagonal (self-similarity)
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
    
    # Compute cross entropy loss
    loss = F.cross_entropy(sim_matrix, labels)
    return loss