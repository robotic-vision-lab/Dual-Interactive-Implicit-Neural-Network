from src.models.sr_module import *
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import save_image
from torchvision import transforms
from torch.nn.functional import avg_pool2d
import torch
import glob
import os
import math
from pathlib import Path
parser = ArgumentParser()
parser.add_argument("--lr_path", type=str, required=True)
parser.add_argument("--output_size", type=int, nargs='+', required=True)
parser.add_argument("--ckpt_path", type=str, required=True)
parser.add_argument("--model_name", type=str, default='default_model')
parser.add_argument("--file_ext", type=str, default='.png')
args = parser.parse_args()

def resize_fn(img, size):
    return transforms.Resize(size=size,
                            interpolation=transforms.InterpolationMode.BICUBIC,
                            antialias=True)(img)

'''
demo2: given an LR image, super-resolve it to the desired resolution
'''
@torch.no_grad()
def demo2(args):
    if args.model_name == 'bicubic':
        model = SRLitModule(arch='bicubic')
    else:
        model = SRLitModule.load_from_checkpoint(args.ckpt_path)

    print(args.lr_path)
    filename, _ = os.path.splitext(os.path.basename(args.lr_path))
    lr = read_image(args.lr_path, ImageReadMode.RGB).unsqueeze(0)/255.
    Path(os.path.dirname(args.lr_path) + "/{}".format(args.model_name)).mkdir(parents=True, exist_ok=True)
    sr = model(lr, args.output_size)
    save_image(sr, os.path.dirname(args.lr_path) + "/{}/{}_{}_{}x{}.png".format(args.model_name, args.model_name, filename, args.output_size[0], args.output_size[1]))

if __name__=='__main__':
    demo2(args)
