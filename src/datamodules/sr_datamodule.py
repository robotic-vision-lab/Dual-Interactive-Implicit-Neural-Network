from typing import Any, Dict, Optional, Tuple, List
import random
from numpy import indices
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.transforms import transforms

from src.datamodules.components.srdata import SRDataDownsample


class Rotation90:
    """Rotate by transpose image."""
    def __call__(self, x):
        if len(x.shape) == 3:
            return x.permute(0,2,1)
        elif len(x.shape) == 4:
            return x.permute(0,1,3,2)
        else:
            print("Something is wrong with Rotation90!!!")

class SRDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        root: str = "./data/",
        trainsets: List[Tuple[str,str]] = [("DIV2K",'train')],
        trainsets_repeat: int = 20,
        testsets: List[Tuple[str,str]] = [("DIV2K",'train'), ('benchmark', 'B100'), ('benchmark', 'Set5'), ('benchmark', 'Set14'), ('benchmark', 'Urban100')],
        batch_size: int = 64,
        train_scales: List[float] = [2,3,4],
        test_scales: List[float] = [2, 2.5, 3, 3.5, 4, 6, 8, 10, 15, 20],
        patch_size: int = 192,
        num_workers: int = 16,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.prepare_data()

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        SRDataDownsample(root=self.hparams.root,
                                name='DIV2K',
                                split='train',
                                scales=self.hparams.train_scales,
                                patch_size=self.hparams.patch_size,
                                augment=False)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_test:
            trainset = []
            for name, split in self.hparams.trainsets:
                if name == 'DIV2K':
                    trainset.append(Subset(SRDataDownsample(root=self.hparams.root,
                                name=name,
                                split=split,
                                scales=self.hparams.train_scales,
                                patch_size=self.hparams.patch_size,
                                augment=True), indices=range(800)))
                else:
                    trainset.append(SRDataDownsample(root=self.hparams.root,
                                name=name,
                                split=split,
                                scales=self.hparams.train_scales,
                                patch_size=self.hparams.patch_size,
                                augment=True))
            trainset = ConcatDataset(trainset)
            self.data_train = ConcatDataset([trainset for _ in range(self.hparams.trainsets_repeat)])

            testset = []
            for name, split in self.hparams.testsets:
                if name == 'DIV2K':
                    testset.append(Subset(SRDataDownsample(root=self.hparams.root,
                                name=name,
                                split=split,
                                scales=self.hparams.test_scales,
                                patch_size=0,
                                augment=False), indices=range(800, 900)))
                else:
                    testset.append(SRDataDownsample(root=self.hparams.root,
                                name=name,
                                split=split,
                                scales=self.hparams.test_scales,
                                patch_size=0,
                                augment=False))
            self.data_test = testset

            self.data_val = Subset(SRDataDownsample(root=self.hparams.root,
                                name='DIV2K',
                                split='train',
                                scales=self.hparams.train_scales,
                                patch_size=0,
                                augment=False), indices=range(800, 900))

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return [DataLoader(
            dataset=data,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        ) for data in self.data_test]

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnist.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
