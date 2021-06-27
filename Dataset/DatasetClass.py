import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset

class CustumDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.targets = target
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = torch.tensor(self.data[idx])
        if self.transform:
            img = self.transform(self.data[idx])
        return img, self.targets[idx]

def GetDataSet(nameData,pathRoot):
    dset = [None,None]
    DATASET_NAME = './'+ nameData
    for i in range(2):
        if nameData == 'cifar100':
            dset[i] = torchvision.datasets.CIFAR100(root=DATASET_NAME,
                                                           train=(i==0),
                                                           transform=transforms.Compose(
                                                               [transforms.ToTensor(),
                                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                                           download=True)
            
 
        elif nameData == 'cifar10':
            dset[i] = torchvision.datasets.CIFAR10(root=DATASET_NAME,
                                                          train=(i==0),
                                                          transform=transforms.Compose(
                                                              [transforms.ToTensor(),
                                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                                          download=True)

        elif nameData == 'SVHN':
            dset[i] = torchvision.datasets.SVHN(root=DATASET_NAME,
                                                       split="train",
                                                       transform=transforms.Compose(
                                                           [transforms.ToTensor(),
                                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                                       download=True)

        elif nameData == 'fashion_mnist':
            dset[i] = torchvision.datasets.FashionMNIST(root=DATASET_NAME,
                                                               train=(i==0),
                                                               transform=transforms.ToTensor(),
                                                               download=True)
        elif nameData == 'mnist':
            dset[i] = torchvision.datasets.MNIST(root=DATASET_NAME,
                                                        train=(i==0),
                                                        transform=transforms.ToTensor(),
                                                        download=True)     
        else:
            print('dataset name is not correct [\'mnist\',\'fashion_mnist\',\'cifar10\',\'cifar100\',\'SVHN\']')
            
    assert (None not in dset)
    return dset[0], dset[1]

