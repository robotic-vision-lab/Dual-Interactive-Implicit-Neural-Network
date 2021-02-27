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

test_data = SRx4Dataset(partition='test')
#test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

subset_indices = [0] # select your indices here as a list

subset = torch.utils.data.Subset(test_data, subset_indices)

test_loader = torch.utils.data.DataLoader(subset, batch_size=1, num_workers=0, shuffle=False)

#model.load_state_dict(torch.load(args.checkpoint_path))
model.eval()

with torch.no_grad():
    for batch in test_loader:
        lr_img, points, img = batch
        lr_img = lr_img.to(device)
        points = points.to(device)
        img = img.to(device)
        out = model(lr_img, points)
        print(out)
        print(out.size())
        #out = model(lr_img, points).reshape(img.shape[0], img.shape[1], img.shape[2], img.shape[3])
        #plt.imshow((out.squeeze(0).permute(1,2,0)*255).int().cpu())
        #plt.show()