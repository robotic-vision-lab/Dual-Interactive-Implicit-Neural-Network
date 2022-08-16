from src.datamodules.sr_datamodule import *
from src.models.sr_module import *
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import save_image
from torchvision import transforms
from torch.nn.functional import avg_pool2d
import glob
import os
parser = ArgumentParser()
parser.add_argument("--scale", type=float)
parser.add_argument("--ckpt_path", type=str)
parser.add_argument("--hr_demo_path", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--file_ext", type=str, default='.png')
args = parser.parse_args()

def resize_fn(img, size):
    return transforms.Resize(size=size,
                            interpolation=transforms.InterpolationMode.BICUBIC,
                            antialias=True)(img)

def demo(args):
    #load image
    names_hr = sorted(glob.glob(os.path.join(str(args.hr_demo_path), '*'+args.file_ext)))
    model = SRLitModule.load_from_checkpoint(args.ckpt_path)
    for f_hr in names_hr:
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        hr = read_image(f_hr, ImageReadMode.RGB).unsqueeze(0)/255.
        lr_size = round(hr.shape[-2]/8), round(hr.shape[-1]/8)
        lr = resize_fn(hr, lr_size)
        save_image(lr, os.path.dirname(f_hr) + "/{}/{}_lr.png".format(args.model_name, filename))
        
        for s in [6, 4, 3, 2, 1]:
            sr = model(lr, (round(hr.shape[-2]/s), round(hr.shape[-1]/s)))
            save_image(avg_pool2d(sr, 3, 1, 1), os.path.dirname(f_hr) + "/{}/{}_x{}.png".format(args.model_name, filename, s))


if __name__=='__main__':
    demo(args)