import os
import argparse
import numpy as np
from metrics import *
import torchvision


parser = argparse.ArgumentParser()
parser.add_argument('--gt_dir', type=str)
parser.add_argument('--pred_dir', type=str)

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]

gt_files = []
pred_files = []

for f in os.listdir(args.gt_dir):
    gt_files.append(os.path.join(args.gt_dir, f))

for f in os.listdir(args.pred_dir):
    pred_files.append(os.path.join(args.pred_dir, f))

gt_files = sorted(gt_files)
pred_files = sorted(pred_files)

#assert len(gt_files) == len(pred_files), "Mismatched number of images!"

print("Number of images: ", len(gt_files))

_PSNR = PSNR()
_SSIM = SSIM()
scores = []
for i in range(len(pred_files)):
    pred = torchvision.io.read_image(pred_files[i]).unsqueeze(0).float()/255
    gt = torchvision.io.read_image(gt_files[i]).unsqueeze(0).float()/255
    psnr = _PSNR(pred, gt)
    ssim = _SSIM(pred, gt)
    scores.append((psnr, ssim))
    print(pred_files[i], gt_files[i], psnr, ssim)

scores = np.array(scores)
print("Average PSNR/SSIM: ", np.average(scores, axis=0))
print("Min PSNR/SSIM: ", np.min(scores, axis=0))
print("Max PSNR/SSIM: ", np.max(scores, axis=0))