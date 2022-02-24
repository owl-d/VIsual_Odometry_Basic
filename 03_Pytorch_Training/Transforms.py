import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(), #converts a PIL image or NumPy ndarray into a FloatTensor
                          #scales the image's pixel intensity values in the range [0, 1]
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
                          #apply any user-difined lambda function
                          #turn the integer into a one-hot encoded tensor
)