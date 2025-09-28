import torch
from torchvision import transforms

class SimCLRDataset(torch.utils.data.Dataset):
    """
    Wrapper dataset that applies two different augmentations to each image for SimCLR
    """
    def __init__(self, base_dataset, transform=None, supervised=False):
        self.base_dataset = base_dataset
        self.transform = transform
        self.supervised = supervised
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get original image and labels
        image, labels = self.base_dataset[idx]

        aug1 = self.transform(image)  # Different each time
        aug2 = self.transform(image)  # Different each time

        if self.supervised:
            return aug1, aug2, labels
        else:
            return aug1, aug2
