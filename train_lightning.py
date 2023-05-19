from pathlib import Path
import lightning.pytorch as pl
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import accuracy
import ovotools.pytorch_tools
import cifar_data
import mnist_data
import model
import params

class TestModule(pl.LightningModule):
    def __init__(self, context):
        super().__init__()
        self.ctx = context
        self.net = context.net
        self.ctx.train_dataloader = None
        self.ctx.val_dataloader = None

    def forward(self, x):
        y = self.net(x)
        return torch.nn.functional.log_softmax(y, dim=1)

    def create_dataloader(self, train):
        if self.ctx.params.data.dataset == 'mnist':
            return mnist_data.mnist_dataloader(train=train, **self.ctx.params.data.params)
        elif self.ctx.params.data.dataset == 'cifar10':
            return cifar_data.cifar10_dataloader(train=train, **self.ctx.params.data.params)

    def train_dataloader(self):
        if self.ctx.train_dataloader is None:
            self.ctx.train_dataloader = self.create_dataloader(train=True)
        return self.ctx.train_dataloader

    def val_dataloader(self):
        if self.ctx.val_dataloader is None:
            self.ctx.val_dataloader = self.create_dataloader(train=False)
        return self.ctx.val_dataloader

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.ctx.loss(logits, y)
        self.log('lr', self.ctx.optimizer.param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('train:loss', loss, prog_bar=True)
        self.log("train:epoch", self.trainer.current_epoch, prog_bar=True)
        self.log("step", self.trainer.fit_loop.epoch_loop._batches_that_stepped*self.ctx.params.data.params.batch_size, prog_bar=True)
        self.log("step", self.trainer.fit_loop.epoch_loop._batches_that_stepped*self.ctx.params.data.params.batch_size, on_step=False, on_epoch=True, prog_bar=True)

        # prob = torch.nn.Softmax()(logits)
        # self.log('train_acc', self.train_acc(prob, y), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss  = self.ctx.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, "multiclass", num_classes=10)
        if stage:
            self.log(f"{stage}:loss", loss, prog_bar=True, logger=True) #, on_step=False, on_epoch=True
            self.log(f"{stage}:acc", acc, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        lr_scheduler = ovotools.pytorch.create_object(self.ctx.params.lr_scheduler, model.eval_func, self.ctx.optimizer)
        return [self.ctx.optimizer], [lr_scheduler]


def train():
    ovotools.pytorch.set_reproducibility()
    pl.seed_everything(42, workers=True)
    ctx = ovotools.pytorch.Context(settings=params.settings, params=params.params, eval_func=model.eval_func)

    # ctx.params.model.params['num_classes'] = len(cifar_data.cifar10_classes)
    if ctx.settings.findLR:
        ctx.params.model_name += '_findLR'
    if ctx.settings.can_overwrite:
        if Path(ctx.params.get_base_filename()).exists():
            shutil.rmtree(ctx.params.get_base_filename())
    ctx.params.save(can_overwrite=ctx.settings.can_overwrite)

    ctx.create_model()
    ctx.create_optim()
    ctx.create_loss()

    debug_params = {
        "limit_train_batches":20,
        "limit_val_batches":10
    } if params.settings.debug_mode else {}

    module = TestModule(ctx)
    trainer = pl.Trainer(
                        logger=pl.loggers.TensorBoardLogger(ctx.params.data_root, name=None, version=params.params.get_model_name() + ("_debug" if params.settings.debug_mode else "")),
                        callbacks=[
                             #pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=True),
                             #pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', period=1, save_top_k = 1, verbose=True)
                        ],
                        max_epochs=ctx.settings.max_epochs,
                        **debug_params
                        )

    trainer.fit(module)




if __name__ == '__main__':
    train()


