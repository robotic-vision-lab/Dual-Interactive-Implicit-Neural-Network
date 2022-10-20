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
parser.add_argument('-d','--d_scale', type=float, required=True)
parser.add_argument('-s','--scales', type=float, nargs='+', required=True)
parser.add_argument("--ckpt_path", type=str, required=True)
parser.add_argument("--model_name", type=str, default='default_model')
parser.add_argument("--file_ext", type=str, default='.png')
args = parser.parse_args()

def resize_fn(img, size):
    return transforms.Resize(size=size,
                            interpolation=transforms.InterpolationMode.BICUBIC,
                            antialias=True)(img)


'''
demo0: downscale each HR image in ./demo/ to obtain corresponding LR images, then super-resolve them at larger and larger scales
'''
@torch.no_grad()
def demo0(args):
    hr_path = "./demo/"
    names_hr = sorted(glob.glob(os.path.join(str(hr_path), '*'+args.file_ext)))
    if args.model_name == 'bicubic':
        model = SRLitModule(arch='bicubic')
    else:
        model = SRLitModule.load_from_checkpoint(args.ckpt_path)
    for f_hr in names_hr:
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        hr = read_image(f_hr, ImageReadMode.RGB).unsqueeze(0)/255.
        lr_size = round(hr.shape[-2]/args.d_scale), round(hr.shape[-1]/args.d_scale)
        lr = resize_fn(hr, lr_size)
        Path(os.path.dirname(f_hr) + "/{}".format(args.model_name)).mkdir(parents=True, exist_ok=True)
        save_image(lr, os.path.dirname(f_hr) + "/{}/{}_lr.png".format(args.model_name, filename))
        for s in args.scales:
            sr = model(lr, (round(hr.shape[-2]/s), round(hr.shape[-1]/s)))
            save_image(sr, os.path.dirname(f_hr) + "/{}/{}_{}x{}.png".format(args.model_name, args.model_name, filename, s))


if __name__=='__main__':
    demo0(args)
