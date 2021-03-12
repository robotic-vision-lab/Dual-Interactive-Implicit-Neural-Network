import model
import trainer
from dataset import SRx4Dataset
import argparse
import torch
from metrics import *
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(
    description='Train Driver'
)

parser.add_argument('--model' , default='Mark_1', type=str)
parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--batch_size' , default=1, type=int)

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]


if torch.cuda.is_available():
    device = torch.device("cuda")
    print('USE CUDA')
else:
    device = torch.device("cpu")
    print('USE CPU')

if args.model == 'Mark_1':
    model = model.Mark_1().to(device)
if args.model == 'Mark_2':
    model = model.Mark_2().to(device)
if args.model == 'Mark_3':
    model = model.Mark_3().to(device)

test_data = SRx4Dataset(partition='test',transform=True)
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
        out, temp = model(lr_img, points)
        #print(out)
        #print(out.size())
        psnr = PSNR(out, img)
        ssim = SSIM(out, img, val_range=255)
        psnr_temp = PSNR(temp, img)
        ssim_temp = SSIM(temp, img, val_range=255)
        f, axarr = plt.subplots(1,4)
        f.suptitle('psnr_pred = {:.4}, ssim_pred = {:.4}, psnr_temp = {:.4}, ssim_temp = {:.4}'.format(psnr, ssim, psnr_temp, ssim_temp))
        axarr[0].imshow(lr_img.squeeze().int().cpu(), cmap='gray', vmin=0, vmax=255)
        axarr[0].set_title('input')
        axarr[1].imshow(out.squeeze().int().cpu(), cmap='gray', vmin=0, vmax=255)
        axarr[1].set_title('pred')
        axarr[2].imshow(img.squeeze().int().cpu(), cmap='gray', vmin=0, vmax=255)
        axarr[2].set_title('gt')
        axarr[3].imshow(temp.squeeze().int().cpu(), cmap='gray', vmin=0, vmax=255)
        axarr[3].set_title('temp')
        plt.show()