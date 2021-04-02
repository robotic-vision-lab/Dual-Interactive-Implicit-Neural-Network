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


print('USE CPU')
device = "cpu"

test_data = DIV2K(root_dir=args.data_dir, partition='valid', downscale_factor=args.scale_factor)
#test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

subset_indices = [3] # select your indices here as a list

subset = torch.utils.data.Subset(test_data, subset_indices)

test_loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)

model.load_state_dict(torch.load(args.checkpoint_path))

model.eval()

with torch.no_grad():
    for batch in test_loader:
        lr_img, points, img = batch
        lr_img = lr_img.to(device)
        points = points.to(device)
        img = img.to(device)
        out = model(lr_img, points)
        #print(out)
        #print(out.size())
        psnr = PSNR(out, img)
        ssim = SSIM(out, img, val_range=255)
        f, axarr = plt.subplots(1,3)
        f.suptitle('psnr_pred = {:.4}, ssim_pred = {:.4}'.format(psnr, ssim))
        axarr[0].imshow(lr_img.squeeze().permute(1,2,0).int())
        axarr[0].set_title('input')
        axarr[1].imshow(out.squeeze().permute(1,2,0).int())
        axarr[1].set_title('pred')
        axarr[2].imshow(img.squeeze().permute(1,2,0).int())
        axarr[2].set_title('gt')
        plt.show()