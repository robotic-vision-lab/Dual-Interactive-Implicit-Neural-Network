from src.datamodules.sr_datamodule import *
from src.models.sr_module import *
from pytorch_lightning.utilities.cli import LightningCLI
import torch
from pytorch_lightning import Trainer
import pdb

if __name__=='__main__':
    cli = LightningCLI(run=False,auto_registry=True, parser_kwargs={"parser_mode": "omegaconf"}, save_config_callback=)
    #pdb.set_trace()