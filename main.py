from src.datamodules.sr_datamodule import *
from src.models.sr_module import *
from pytorch_lightning.utilities.cli import LightningCLI

if __name__=='__main__':
    cli = LightningCLI(run=False, auto_registry=True, parser_kwargs={"parser_mode": "omegaconf"})
    dm=cli.datamodule
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loaders = dm.test_dataloader()
    print(len(train_loader.dataset))
    print(len(val_loader.dataset))
    print([len(test_loader.dataset) for test_loader in test_loaders])

    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
    ckpt_path = cli.trainer.checkpoint_callback.last_model_path
    if ckpt_path == "":
        ckpt_path = None
    cli.trainer.test(model=cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)