import torch
import torchvision
import torchvision.transforms as transforms
import local_config

#data
# from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def cifar10_dataset(train):
    dataset = torchvision.datasets.CIFAR10(root=local_config.data_root, train=train, download=True,
                                            transform=transform_train if train else transform_test)
    return dataset

def cifar10_dataloader(train, batch_size):
    dataset = cifar10_dataset(train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0)
    return loader


