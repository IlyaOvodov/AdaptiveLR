from collections import OrderedDict
import ignite
from ignite.engine import Events
import ovotools.ignite_tools
import ovotools.pytorch_tools
import cifar_data
import mnist_data
import model

import params

def train():
    ctx = ovotools.pytorch.Context(settings=params.settings, params=params.params, eval_func=model.eval_func)

    ctx.params.model.params['num_classes'] = len(cifar_data.cifar10_classes)
    if ctx.settings.findLR:
        ctx.params.model_name += '_findLR'
    ctx.params.save(can_overwrite = ctx.settings.can_overwrite)

    # Reproducability
    ovotools.pytorch.set_reproducibility()

    ctx.create_model()
    ctx.create_optim()
    ctx.create_loss()

    if ctx.params.data.dataset == 'mnist':
        data_loader = mnist_data.mnist_dataloader(train=True, **ctx.params.data.params)
        val_data_loader = mnist_data.mnist_dataloader(train=False, **ctx.params.data.params)
    elif ctx.params.data.dataset == 'cifar10':
        data_loader = cifar_data.cifar10_dataloader(train = True, **ctx.params.data.params)
        val_data_loader = cifar_data.cifar10_dataloader(train = False, **ctx.params.data.params)

    trainer_metrics = OrderedDict({
        'loss': ignite.metrics.Loss(ctx.loss.get_val(), batch_size=lambda y: ctx.params.data.batch_size),
        'accuracy': ignite.metrics.Accuracy(),
    })

    trainer = ovotools.ignite_tools.create_supervised_trainer(ctx.net, ctx.optimizer, ctx.loss, metrics=trainer_metrics, device = ctx.settings.device)
    evaluator = ignite.engine.create_supervised_evaluator(ctx.net, metrics=trainer_metrics, device = ctx.settings.device)

    if ctx.settings.findLR:
        best_model_buffer = None
    else:
        best_model_buffer = ovotools.ignite_tools.BestModelBuffer(ctx.net, 'val:loss', minimize = True, params = ctx.params)
    log_training_results = ovotools.ignite_tools.LogTrainingResults(evaluator = evaluator,
                                                                    loaders_dict = {'val':val_data_loader},
                                                                    best_model_buffer = best_model_buffer,
                                                                    params = ctx.params,
                                                                    duty_cycles = 1)
    eval_event = ignite.engine.Events.ITERATION_COMPLETED if ctx.settings.findLR else ignite.engine.Events.EPOCH_COMPLETED
    trainer.add_event_handler(eval_event, log_training_results, event = eval_event)

    tb_logger = ovotools.ignite_tools.TensorBoardLogger(trainer,ctx.params,count_iters = ctx.settings.findLR)
    tb_logger.start_server(ctx.settings.tensorboard_port, start_it = False)

    timer = ovotools.ignite_tools.IgniteTimes(trainer, count_iters = False, measured_events = {
        'train:time.iter': (trainer, Events.ITERATION_STARTED, Events.ITERATION_COMPLETED),
        'train:time.epoch': (trainer, Events.EPOCH_STARTED, Events.EPOCH_COMPLETED),
        'val:time.epoch': (evaluator, Events.EPOCH_STARTED, Events.EPOCH_COMPLETED),
    })

    if ctx.settings.findLR:
        import math
        @trainer.on(Events.ITERATION_STARTED)
        def upd_lr(engine):
            log_lr = ctx.params.lr_finder.log_lr_start + (ctx.params.lr_finder.log_lr_end - ctx.params.lr_finder.log_lr_start) * (engine.state.iteration-1)/ctx.params.lr_finder.iters_num
            lr = math.pow(10, log_lr)
            ctx.optimizer.param_groups[0]['lr'] = lr
            engine.state.metrics['lr'] = ctx.optimizer.param_groups[0]['lr']
            if engine.state.iteration > ctx.params.lr_finder.iters_num:
                print('done')
                engine.terminate()
    else:
        #clr_scheduler = ovotools.ignite_tools.ClrScheduler(data_loader, ctx.net, ctx.optimizer, 'train:loss', ctx.params, engine=trainer)
        ctx.lr_scheduler = ovotools.pytorch.create_optional_object(ctx.params, 'lr_scheduler', ctx.eval_func, optimizer=ctx.optimizer)
        @trainer.on(Events.EPOCH_COMPLETED)
        def lr_scheduler_step(engine):
            engine.state.metrics['lr'] = ctx.optimizer.param_groups[0]['lr']
            if ctx.lr_scheduler:
                call_params = {'epoch': engine.state.epoch}
                if ctx.params.lr_scheduler.type.split('.')[-1] == 'ReduceLROnPlateau':
                    call_params['metrics'] = engine.state.metrics['val:loss']
                ctx.lr_scheduler.step(**call_params)


    @trainer.on(Events.ITERATION_COMPLETED)
    def reset_resources(engine):
        engine.state.batch = None
        engine.state.output = None
        #torch.cuda.empty_cache()

    trainer.run(data_loader, max_epochs = ctx.settings.max_epochs)

if __name__ == '__main__':
    train()





