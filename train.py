from src.datamodules.sr_datamodule import *
from src.models.sr_module import *
import torch
from pytorch_lightning import Trainer
import pdb
import hydra
from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="configs", config_name="hydra_config")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    datamodule = instantiate(cfg.data)
    trainer = instantiate(cfg.trainer)
    model = instantiate(cfg.model)

    trainer.fit(model=model, datamodule=datamodule)

if __name__ == "__main__":
    train()