import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_datas(batch_size):
    train_transform = transforms.Compose([
        #transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        #transforms.RandomErasing(p=0.5, scale=(0.02, 0.1))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_set = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    return train_loader, test_loader
