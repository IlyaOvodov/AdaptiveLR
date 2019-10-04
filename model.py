import os
import torch
import torchvision
import local_config

def create_model(params, train = True):
    model_class = eval(params.model)
    model = model_class(**params.model_params)
    model.train(train)
    return model

def create_lr_scheduler(optimizer, params):
    lrsched_class = eval(params.ls_cheduler)
    params_name = params.ls_cheduler.split('.')[-1] + '_params'
    assert params_name in params, params_name + ' must be in params dict()'
    lrsched_params = params[params_name]
    lrsched = lrsched_class(optimizer, **lrsched_params)
    return lrsched
