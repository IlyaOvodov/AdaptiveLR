from ovotools import AttrDict
import local_config

# Parameters of program run, not of algorithm. Not expected to imfluence the result
settings = AttrDict(
    max_epochs = 200,
    tensorboard_port = 6006,
    device = 'cuda:0',
    findLR = False,
    can_overwrite = True,
)

# Parameters of algorithm. Are stored with result
params = AttrDict(
    data_root = local_config.data_root,
    model_name='NN_results/llr',
    data = AttrDict(
        dataset = 'mnist',
        batch_size = 128,
    ),
    model = AttrDict(
        type = 'MnistNet',
        params = AttrDict(),
        #load_from = 'NN_results/segmentation/linknet_128x128_cc60ab/models/clr.003.t7',
    ),
    loss=AttrDict(
        type='torch.nn.CrossEntropyLoss',
    ),
    optimizer = AttrDict(
        type = 'llr.LLR_SGD', #'torch.optim.SGD' 'binary_optymizer.SGD_binary'
        params = AttrDict(
            lr=0.001,
            momentum=.5,
            weight_decay = 1e-3,
            # nesterov = False,
            lrlr = 0.01,
            update_mode = 'loggrad', # const: lr */= (1+lrlr) loggrad
        ),
    ),
    lr_finder = AttrDict(
        iters_num=200,
        log_lr_start=-6,
        log_lr_end=-1,
    ),
    lr_scheduler = AttrDict(
        #type = 'torch.optim.lr_scheduler.ReduceLROnPlateau',
        params=AttrDict(
            mode='min',
            factor=0.5,
            patience=10,
            #threshold=0.01
        ),
    ),
)
