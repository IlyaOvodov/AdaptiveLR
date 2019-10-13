import torch
from torchvision import datasets, transforms
import local_config
import ovotools.pytorch

def mnist_dataloader(params, train):
    dataset = datasets.MNIST(root=local_config.data_root, train=train, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

    dataset = ovotools.pytorch.CachedDataSet(dataset)

    loader = torch.utils.data.DataLoader( dataset, batch_size=params.data.batch_size, shuffle=train, num_workers=0)
    #loader = ovotools.pytorch.BatchThreadingDataLoader( dataset, batch_size=params.data.batch_size, shuffle=train, num_workers=10)

    return loader