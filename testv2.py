from argparse import ArgumentParser
import torch
from src.models.components.imsisr import IMSISR
from src.datamodules.components.srdata import SRDataDownsample
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
parser = ArgumentParser()
parser.add_argument("--ckpt_path", type=str)
parser.add_argument("--name", type=str)
parser.add_argument("--split", type=str)
parser.add_argument("--scale", type=int)
args = parser.parse_args()

def test(args):
    checkpoint = torch.load(args.ckpt_path)
    model = IMSISR(mode=checkpoint['hyper_parameters']['mode'], init_q=checkpoint['hyper_parameters']['init_q'])
    model_weights = {}
    for k, v in checkpoint['state_dict'].items():
        if k.find('net.') > -1:
            model_weights[k[4:]] = v
    net.load_state_dict(model_weights)
    
    dataset = SRDataDownsample(root="./data/",
                                name=args.name,
                                split=args.split,
                                scales=[args.scale],
                                patch_size=0,
                                augment=False)
    
    loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    

if __name__=='__main__':
    test(args)