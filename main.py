from src.datamodules.sr_datamodule import *
from src.models.sr_module import *
from pytorch_lightning.utilities.cli import LightningCLI
import torch
from pytorch_lightning import Trainer

if __name__=='__main__':
    cli = LightningCLI(run=False, auto_registry=True, parser_kwargs={"parser_mode": "omegaconf"})

    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)


    #test
    print("Testing!!!")
    ckpt_path = cli.trainer.checkpoint_callback.last_model_path
    if ckpt_path == "":
        ckpt_path = None
    test_config = cli.config_init.trainer
    test_config.devices=1
    test_trainer = Trainer(**test_config)
    test_trainer.test(model=cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)