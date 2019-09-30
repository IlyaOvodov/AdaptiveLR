import torch
import torchvision
import torchvision.transforms as transforms
from ovotools import AttrDict
import local_config
import cifar_data

params = AttrDict(
    data_root = local_config.data_root,
    model_name='NN_results/base',
    data = AttrDict(
        batch_size = 128,
    ),
    model='torchvision.models.resnet18',
    model_params=AttrDict(
        num_classes= len(cifar_data.cifar10_classes)
    ),
    #model_load_from = 'NN_results/segmentation/linknet_128x128_cc60ab/models/clr.003.t7',
    optim='torch.optim.SGD',
    optim_params=AttrDict(
        lr=0.1,
        momentum=0.9,
        weight_decay = 5e-4,
        # nesterov = False,
    ),
    lr_finder = AttrDict(
        iters_num=400,
        log_lr_start=-4,
        log_lr_end=-0,
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

max_epochs = 350
tensorboard_port = 6006
device = 'cuda:0'
findLR = False
can_overwrite = False
max_data_len = None

if findLR:
    params.model_name += '_findLR'
params.save(can_overwrite = can_overwrite)

from collections import OrderedDict
import torch
import torch.nn.functional as F
import ignite
from ignite.engine import Events
import ovotools.ignite_tools
import ovotools.pytorch_tools
import model

net = model.create_model(params).to('cuda')
data_loader = cifar_data.cifar10_dataloader(params, train = True)
val_data_loader = cifar_data.cifar10_dataloader(params, train = False)
optimizer = eval(params.optim)(net.parameters(), **params.optim_params)
loss = torch.nn.CrossEntropyLoss()

trainer_metrics = OrderedDict({
    'loss': ignite.metrics.Loss(loss, batch_size=lambda y: params.data.batch_size),
    'accuracy': ignite.metrics.Accuracy(),
})

train_epochs = params.lr_finder.iters_num*len(data_loader) if findLR else max_epochs

trainer = ovotools.ignite_tools.create_supervised_trainer(net, optimizer, loss, metrics=trainer_metrics, device = device)
evaluator = ignite.engine.create_supervised_evaluator(net, metrics=trainer_metrics, device = device)

if findLR:
    best_model_buffer = None
else:
    best_model_buffer = ovotools.ignite_tools.BestModelBuffer(net, 'train:accuracy', minimize = False, params = params)
log_training_results = ovotools.ignite_tools.LogTrainingResults(evaluator = evaluator,
                                                                loaders_dict = {'val':val_data_loader},
                                                                best_model_buffer = best_model_buffer,
                                                                params = params,
                                                                duty_cycles = 1)
eval_event = ignite.engine.Events.ITERATION_COMPLETED if findLR else ignite.engine.Events.EPOCH_COMPLETED
trainer.add_event_handler(eval_event, log_training_results, event = eval_event)

tb_logger = ovotools.ignite_tools.TensorBoardLogger(trainer,params,count_iters = findLR)
tb_logger.start_server(tensorboard_port, start_it = False)

timer = ovotools.ignite_tools.IgniteTimes(trainer, count_iters = False, measured_events = {
    'train:time.iter': (trainer, Events.ITERATION_STARTED, Events.ITERATION_COMPLETED),
    'train:time.epoch': (trainer, Events.EPOCH_STARTED, Events.EPOCH_COMPLETED),
    'val:time.epoch': (evaluator, Events.EPOCH_STARTED, Events.EPOCH_COMPLETED),
})

if findLR:
    import math
    @trainer.on(Events.ITERATION_STARTED)
    def upd_lr(engine):
        log_lr = params.lr_finder.log_lr_start + (params.lr_finder.log_lr_end - params.lr_finder.log_lr_start) * (engine.state.iteration-1)/params.lr_finder.iters_num
        lr = math.pow(10, log_lr)
        optimizer.param_groups[0]['lr'] = lr
        engine.state.metrics['lr'] = optimizer.param_groups[0]['lr']
        if engine.state.iteration > params.lr_finder.iters_num:
            print('done')
            engine.terminate()
else:
    #clr_scheduler = ovotools.ignite_tools.ClrScheduler(data_loader, net, optimizer, 'train:loss', params, engine=trainer)
    lr_scheduler = model.create_lr_scheduler(optimizer, params)

    @trainer.on(Events.EPOCH_COMPLETED)
    def lr_scheduler_step(engine):
        call_params = {'epoch': engine.state.epoch}
        if params.ls_cheduler.split('.')[-1] == 'ReduceLROnPlateau':
            call_params['metrics'] = engine.state.metrics['val:loss']
        engine.state.metrics['lr'] = optimizer.param_groups[0]['lr']
        lr_scheduler.step(**call_params)

@trainer.on(Events.ITERATION_COMPLETED)
def reset_resources(engine):
    engine.state.batch = None
    engine.state.output = None
    #torch.cuda.empty_cache()

train_epochs = params.lr_finder.iters_num*len(data_loader) if findLR else max_epochs
trainer.run(data_loader, max_epochs = max_epochs)







