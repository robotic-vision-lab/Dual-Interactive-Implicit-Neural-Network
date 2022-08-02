from src.datamodules.sr_datamodule import *
from src.models.sr_module import *
from pytorch_lightning.utilities.cli import LightningCLI
import torch

if __name__=='__main__':
    cli = LightningCLI(run=False, auto_registry=True, parser_kwargs={"parser_mode": "omegaconf"})
    #print(cli.config_init)
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)


    #test
    ckpt_path = cli.trainer.checkpoint_callback.last_model_path
    if ckpt_path == "":
        ckpt_path = None
    cli.trainer.test(model=cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)