import re
from typing import Any, List
from functools import partial
import torch
from pytorch_lightning import LightningModule
from src.models.components.liif import LIIF
from src.models.components.metasr import MetaSR
from src.models.components.imsisr import IMSISR
from torchmetrics import MaxMetric, PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchvision import transforms

def resize_fn(img, size):
    return transforms.Resize(size=size,
                            interpolation=transforms.InterpolationMode.BICUBIC,
                            antialias=True)(img)

def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = int(scale)
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = int(scale) + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)



def make_net(arch, mode, init_q):
    if arch == 'liif':
        return LIIF()
    elif arch == 'metasr':
        return MetaSR()
    elif arch == 'imsisr':
        return IMSISR(mode=mode, init_q=init_q)
    

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
        arch: str,
        mode: int = 1,
        init_q: bool = False,
        lr: float = 1e-4,
        lr_gamma: float = 0.5,
        lr_step: int = 10,
        eval_bsize: int = 30000
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = make_net(self.hparams.arch, self.hparams.mode, self.hparams.init_q)

        #data norm
        self.register_buffer("sub", torch.FloatTensor([0.5]).view(1, -1, 1, 1))
        self.register_buffer("div", torch.FloatTensor([0.5]).view(1, -1, 1, 1))
        # loss function
        self.criterion = torch.nn.L1Loss()

        # for logging best so far validation accuracy
        #self.val_psnr_best = MaxMetric()

    def forward(self, x: torch.Tensor, size, eval_bsize=None):
        return self.net(x, size, eval_bsize)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        #self.val_psnr_best.reset()
        pass

    def step(self, batch: Any, eval_bsize=None):
        loss = 0
        pred_hrs = {}
        #hrs = {}
        for scale in batch:
            lr, hr, _ = batch[scale]
            lr = (lr - self.sub) / self.div
            hr = (hr - self.sub) / self.div
            pred_hr = self.forward(lr, [round(lr.shape[-2]*scale), round(lr.shape[-1]*scale)], eval_bsize)
            loss += self.criterion(pred_hr, hr)
            pred_hrs[scale] = (pred_hr * self.div + self.sub).clamp_(0, 1)
            #hrs[scale] = hr
        return loss/len(batch), pred_hrs

    def training_step(self, batch: Any, batch_idx: int):
        loss, _= self.step(batch)

        B = len(batch) * batch[2][0].shape[0]
        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=B)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, pred_hrs = self.step(batch, self.hparams.eval_bsize)

        B = len(batch) * batch[2][0].shape[0]
        # log val metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=B)
        
        for scale in batch:
            psnr_func = partial(calc_psnr, dataset='div2k', scale=scale, rgb_range=1)
            psnr = psnr_func(pred_hrs[scale], batch[scale][1])
            self.log("val/psnr_x{}".format(scale), psnr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=B/len(batch))
        return {}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int):
        _, pred_hrs = self.step(batch, self.hparams.eval_bsize)
        
        res = {}
        for scale in batch:
            #compute psnr
            res['psnr_res'] = psnr(pred_hrs[scale], batch[scale][1], data_range=255)
            #compute ssim
            res['ssim_res'] = ssim(pred_hrs[scale], batch[scale][1], data_range=255)
            #compute lr_psnr
            lr_h = round(batch[scale][1].shape[-2] / scale) 
            lr_w = round(batch[scale][1].shape[-1] / scale)
            lr_pred = resize_fn(pred_hrs[scale], (lr_h, lr_w))
            lr_target = resize_fn(batch[scale][1], (lr_h, lr_w))
            res['lr_psnr_res'] = psnr(lr_pred, lr_target, data_range=255)
            #logging
            self.log("test/psnr_x{}".format(scale), res['psnr_res'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=1)
            self.log("test/ssim_x{}".format(scale), res['ssim_res'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=1)
            self.log("test/lr_psnr_x{}".format(scale), res['lr_psnr_res'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=1)
        return res

    def test_epoch_end(self, outputs: List[Any]):
        print(outputs)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.hparams.lr_step, gamma=self.hparams.lr_gamma)
        return [optimizer], [scheduler]

