import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset

class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir_inp, rgb_dir_tar, img_options=None, transform= None): 
        super(DataLoaderTrain, self).__init__()
        self.inp_filenames = rgb_dir_inp #.npy 확장자 path
        self.tar_filenames = rgb_dir_tar #.npy 확장자 path

        self.img_options = img_options
        self.sizex= len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

        self.transform = transform
        
    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_img = np.load(self.inp_filenames[index_], allow_pickle=True)
        tar_img = np.load(self.tar_filenames[index_], allow_pickle=True)
        
#         inp_img = self.transform(image=inp_img)['image']
        inp_img = TF.to_tensor(inp_img)
        return tar_img, inp_img
    
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

