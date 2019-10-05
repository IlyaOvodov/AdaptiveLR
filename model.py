import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import local_config
import binary_optymizer

def create_object(params, key, required,  *args, **kwargs):
    p = params.get(key, '')
    if not p:
        if required:
            raise Exception('NO '+ key + ' is set')
        else:
            print('NO '+ key + ' is set')
            return None
    all_args = kwargs.copy()
    all_args.update(dict(p.get('params', dict())))
    print('creating: ', p.type, all_args)
    obj = eval(p.type)(*args, **all_args)
    return obj

def create_model(params, train = True):
    net = create_object(params, 'model', True)
    net.train(train)
    return net

def create_optim(net_parameters, params):
    return create_object(params, 'optim', True, net_parameters)

def create_lr_scheduler(optimizer, params):
    return create_object(params, 'lr_cheduler', False, optimizer)

'''
def create_model(params, train = True):
    print('creating: ', params.model.type, dict(params.model.get('params', dict())))
    net = eval(params.model.type)(**params.model.get('params', dict()))
    net.train(train)
    return net

def create_optim(net, params):
    print('creating: ', params.optim.type, dict(params.optim.get('params', dict())))
    optimizer = eval(params.optim.type)(net.parameters(), **params.optim.get('params', dict()))
    return optimizer

def create_lr_scheduler(optimizer, params):
    if not params.get('lr_cheduler', ''):
        print('NO lr_cheduler set')
        return None
    print('creating: ', params.lr_cheduler.type, dict(params.lr_cheduler.get('params', dict())))
    lrsched = eval(params.lr_cheduler.type)(optimizer, **params.lr_cheduler.get('params', dict()))
    return lrsched
'''

class MnistNet(nn.Module):
    def __init__(self, **kwargs):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x #F.log_softmax(x, dim=1)