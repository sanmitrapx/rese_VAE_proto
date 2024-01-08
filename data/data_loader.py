import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import random_split, Dataset
from torch.utils.data.dataloader import DataLoader

class Binarizer(Dataset):
    """
    Convert the greyscale image to binary pixel values
    """
    def __init__(self, ds, threshold=0.5):
        """
        ds: Dataset
        threshold: pixel values>threshold -> 1,
        0 otherwise
        """
        self._ds = ds
        self._threshold = threshold

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        x, y = self._ds[idx]
        return (x >= self._threshold).float(), y
    
def setup_data_loaders(batch_size=128, bin_data = True):
    train_dataset = FashionMNIST(root='data/', train=True, download=True,
               transform=transforms.Compose([transforms.Resize(64), transforms.ToTensor()]))
    test_dataset = dataset = FashionMNIST(root='data/', train=False, download=True,
               transform=transforms.Compose([transforms.Resize(64), transforms.ToTensor()]))
    val_size = 5000
    train_size = len(train_dataset) - val_size
    train_ds, val_ds = random_split(train_dataset, [train_size, val_size])
    
    
    if bin_data:
        train_ds = Binarizer(train_ds)
        val_ds = Binarizer(val_ds)
        test_ds = Binarizer(test_dataset)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, pin_memory=False)
    test_loader = DataLoader(val_ds, batch_size, shuffle=True, pin_memory=False)
    return train_loader, val_loader, test_loader    