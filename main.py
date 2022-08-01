from src.datamodules.sr_datamodule import *
from src.models.sr_module import *
from pytorch_lightning.utilities.cli import LightningCLI

if __name__=='__main__':
    LightningCLI(auto_registry=True, parser_kwargs={"parser_mode": "omegaconf"})