from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, PeakSignalNoiseRatio



class SRLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        lr: float
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # loss function
        self.criterion = torch.nn.L1Loss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.test_psnr = PeakSignalNoiseRatio(data_range=1)

        # for logging best so far validation accuracy
        #self.val_psnr_best = MaxMetric()

    def quantize(self, img):
        return img.mul(255).clamp(0, 255).round().div(255) 

    def forward(self, x: torch.Tensor, scale):
        return self.net(x, scale)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        #self.val_psnr_best.reset()
        pass

    def step(self, batch: Any):
        loss = 0
        pred_hrs = {}
        hrs = {}
        for scale in batch:
            lr, hr, _ = batch[scale]
            pred_hr = self.forward(lr, scale)
            loss += self.criterion(pred_hr, hr)
            pred_hrs[scale] = pred_hr
            hrs[scale] = hr
        return loss, pred_hrs, hrs

    def training_step(self, batch: Any, batch_idx: int):
        loss, pred_hrs, hrs = self.step(batch)

        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        for scale in batch:
            psnr = self.train_psnr(self.quantize(pred_hrs[scale]), hrs[scale])
            self.log("train/psnr_x{}".format(scale), psnr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_psnr.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, pred_hrs, hrs = self.step(batch)

        # log val metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        for scale in batch:
            psnr = self.val_psnr(self.quantize(pred_hrs[scale]), hrs[scale])
            self.log("val/psnr_x{}".format(scale), psnr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        self.val_psnr.reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, pred_hrs, hrs = self.step(batch)

        # log test metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        for scale in batch:
            psnr = self.val_psnr(self.quantize(pred_hrs[scale]), hrs[scale])
            self.log("test/psnr_x{}".format(scale), psnr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        self.test_psnr.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.5)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
