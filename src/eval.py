from model import Mark_2, Mark_3, Mark_4
from dataset import DIV2K
import argparse
import torch
from metrics import *
import matplotlib.pyplot as plt
import os
import torchvision
parser = argparse.ArgumentParser(
    description='Eval Driver'
)

parser.add_argument('--model' , default='Mark_1', type=str)
parser.add_argument('--scale_factor', type=int)
parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--data_dir', type=str)

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]


if __name__ == '__main__':
    device = 'cpu'
    if args.model == 'Mark_1':
        model = Mark_1().to(device)
    elif args.model == 'Mark_2':
        model = Mark_2().to(device)
    elif args.model == 'Mark_3':
        model = Mark_3().to(device)
    elif args.model == 'Mark_4':
        model = Mark_4().to(device)

    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()

    test_data = DIV2K(root_dir=args.data_dir, partition='valid', downscale_factor=args.scale_factor, eval=True)
    #test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    #subset_indices = torch.randint(0, len(test_data), (10,)) # select your indices here as a list

    #subset = torch.utils.data.Subset(test_data, subset_indices)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=None, shuffle=False)



    save_path = os.path.join(os.path.dirname(args.checkpoint_path), 'x'+str(args.scale_factor))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        for f in os.listdir(save_path):
            os.remove(os.path.join(save_path, f))
    print(save_path)

    with torch.no_grad():
        for batch in test_loader:
            lr_img, points, filename = batch
            print(lr_img.size(), points.size(), filename)
            #lr_imgs = lr_imgs.to(device)
            #points = points.to(device)
            #outs = model(lr_imgs, points)

            
            lr_img = lr_img.to(device)
            code = model.encoder(lr_img)
            s = points.shape[2]
            out = []
            d = 10
            for i in range(d):
                p = points[:, :, i*s//d : (i+1)*s//d, :].to(device)
                temp = model.decoder(code, p.reshape((p.shape[0], -1, 1, p.shape[3])))
                out.append(temp.cpu().reshape((p.shape[0], 3, p.shape[1], p.shape[2])))
            out = torch.cat(out, dim=3)


            torchvision.utils.save_image(out.squeeze(), os.path.join(save_path, filename))
            
            #print(out)
            #print(outs.size())
            #imgs = imgs.to(device)
            #f, axarr = plt.subplots(3)
            #psnr = PSNR()(outs, imgs)
            #ssim = SSIM()(outs, imgs)
            #axarr[0].imshow(lr_imgs[0].squeeze().permute(1,2,0))
            #axarr[1].imshow(outs[0].squeeze().permute(1,2,0))
            #axarr[2].imshow(imgs[0].squeeze().permute(1,2,0))
            #axarr[0].set_title('input')
            #axarr[1].set_title('pred')
            #axarr[2].set_title('gt')
            #f.suptitle('psnr_pred = {:.4}, ssim_pred = {:.4}'.format(psnr, ssim))
            #plt.show()