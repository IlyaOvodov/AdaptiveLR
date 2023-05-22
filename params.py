from ovotools import AttrDict
import local_config


# Parameters of program run, not of algorithm. Not expected to imfluence the result
settings = AttrDict(
    debug_mode = False,
    max_epochs = 10,
    tensorboard_port = 6006,
    device = 'cuda:0',
    findLR = False,
    can_overwrite = True,
)

# Parameters of algorithm. Are stored with result
params = AttrDict(
    data_root = local_config.data_root,
    model_name='NN_results/2023/{data.dataset}loop_test/04_separate_{model.type}_bs{data.params.batch_size}_{optimizer.type}_{optimizer.params.lr}',
    data = AttrDict(
        dataset = 'mnist',
        params = AttrDict(
            batch_size = 100,
        )
    ),
    model = AttrDict(
        type = 'LeNet',
        params = AttrDict(),
        #load_from = 'NN_results/segmentation/linknet_128x128_cc60ab/models/clr.003.t7',
    ),
    loss=AttrDict(
        type='torch.nn.NLLLoss',
        params=AttrDict(
            reduction='none',
        ),
    ),
    optimizer = AttrDict(
        type = 'SGD', # 'torch.optim.SGD' 'llr.LLR_SGD' 'binary_optymizer.SGD_binary'
        params = AttrDict(
            lr=0.1,
            momentum=.5,
            weight_decay = 1e-3,
            # nesterov = False,
            # lrlr = 0.01,
            # update_mode = 'loggrad', # const: lr */= (1+lrlr) loggrad
            # use = 'd_p', # 'grad', 'd_p'
        ),
    ),
    lr_finder = AttrDict(
        iters_num=200,
        log_lr_start=-2,
        log_lr_end=-2,
    ),
    lr_scheduler = AttrDict(
        type = 'torch.optim.lr_scheduler.StepLR',
        params=AttrDict(
            step_size=5,
            gamma=0.1,
        ),
    ),
)
