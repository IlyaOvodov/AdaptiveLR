from ovotools import AttrDict
import local_config

params = AttrDict(
    data_root = local_config.data_root,
    model_name='NN_results/repr3',
    data = AttrDict(
        dataset = 'mnist',
        batch_size = 128,
    ),
    model = AttrDict(
        type = 'MnistNet',
        params = AttrDict(),
        #load_from = 'NN_results/segmentation/linknet_128x128_cc60ab/models/clr.003.t7',
    ),
    optim = AttrDict(
        type = 'torch.optim.SGD', #'torch.optim.SGD' 'binary_optymizer.SGD_binary'
        params = AttrDict(
            lr=0.01,
            momentum=.5,
            #weight_decay = 1e-3,
            # nesterov = False,
        ),
    ),
    lr_finder = AttrDict(
        iters_num=200,
        log_lr_start=-6,
        log_lr_end=-1,
    ),
    lr_cheduler = AttrDict(
        #type = 'torch.optim.lr_scheduler.ReduceLROnPlateau',
        params=AttrDict(
            mode='min',
            factor=0.5,
            patience=10,
            #threshold=0.01
        ),
    ),
)
