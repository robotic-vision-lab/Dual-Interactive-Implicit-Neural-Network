from src.datamodules.sr_datamodule import *
from src.models.sr_module import *
from pytorch_lightning import Trainer
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--ckpt_path", type=str)
args = parser.parse_args()

def test(args):
    model = SRLitModule.load_from_checkpoint(args.ckpt_path)
    datamodule = SRDataModule()
    trainer = Trainer(accelerator='gpu', devices=1)
    trainer.test(model=model, datamodule=datamodule)

if __name__=='__main__':
    test(args)