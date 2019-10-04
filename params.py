from ovotools import AttrDict
import local_config

params = AttrDict(
    data_root = local_config.data_root,
    model_name='NN_results/base',
    data = AttrDict(
        batch_size = 128,
    ),
    model='torchvision.models.resnet18',
    model_params=AttrDict(
    ),
    #model_load_from = 'NN_results/segmentation/linknet_128x128_cc60ab/models/clr.003.t7',
    optim='torch.optim.SGD', #'torch.optim.SGD' 'binary_optymizer.SGD_binary'
    optim_params=AttrDict(
        lr=2e-5,
        momentum=0.9,
        weight_decay = 5e-4,
        # nesterov = False,
    ),
    lr_finder = AttrDict(
        iters_num=200,
        log_lr_start=-5.5,
        log_lr_end=-4,
    ),
    ls_cheduler = 'torch.optim.lr_scheduler.MultiStepLR',
    clr=AttrDict(
        warmup_epochs=1,
        min_lr=2e-4,
        max_lr=3e-1,
        period_epochs=20,
        scale_max_lr=0.95,
        scale_min_lr=0.95,
    ),
    ReduceLROnPlateau_params=AttrDict(
        mode='min',
        factor=0.1,
        patience=10,
        threshold=0.01
    ),
    MultiStepLR_params=AttrDict(
        milestones = [150, 250],
        gamma = 0.1
    )
)
