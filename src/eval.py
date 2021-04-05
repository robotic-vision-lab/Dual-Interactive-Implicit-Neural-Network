import model
from dataset import DIV2K
import argparse
import torch
from metrics import *
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(
    description='Train Driver'
)

parser.add_argument('--model' , default='Mark_1', type=str)
parser.add_argument('--scale_factor' , default=4, type=int)
parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--batch_size' , default=1, type=int)
parser.add_argument('--data_dir', type=str)

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]

if args.model == 'Mark_1':
    model = model.Mark_1()

if torch.cuda.is_available() and False:
    print('USE CUDA')
    device = "cuda"
else:
    print('USE CPU')
    device = "cpu"

test_data = DIV2K(root_dir=args.data_dir, partition='valid', downscale_factor=args.scale_factor, eval=True)
#test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

subset_indices = torch.randint(0, len(test_data), (3,)) # select your indices here as a list

subset = torch.utils.data.Subset(test_data, subset_indices)

test_loader = torch.utils.data.DataLoader(subset, batch_size=None, shuffle=False)

model.load_state_dict(torch.load(args.checkpoint_path))
model.to(device)
model.eval()

with torch.no_grad():
    for batch in test_loader:
        lr_imgs, points, imgs = batch
        lr_imgs = lr_imgs.to(device)
        points = points.to(device)
        print(lr_imgs.size(), points.size())
        outs = model(lr_imgs, points)
        #print(out)
        #print(outs.size())
        imgs = imgs.to(device)
        f, axarr = plt.subplots(5,3)
        psnr = 0.0
        ssim = 0.0
        for i in range(5):
            psnr += PSNR(outs[i].unsqueeze(0), imgs[i].unsqueeze(0))
            ssim += SSIM(outs[i].unsqueeze(0), imgs[i].unsqueeze(0), val_range=255)
            axarr[i, 0].imshow(lr_imgs[i].squeeze().permute(1,2,0).int())
            axarr[i, 1].imshow(outs[i].squeeze().permute(1,2,0).int())
            axarr[i, 2].imshow(imgs[i].squeeze().permute(1,2,0).int())
        axarr[0, 0].set_title('input')
        axarr[0, 1].set_title('pred')
        axarr[0, 2].set_title('gt')
        f.suptitle('psnr_pred = {:.4}, ssim_pred = {:.4}'.format(psnr/5, ssim/5))
        plt.show()