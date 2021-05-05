import os
from model import Mark_1, Mark_2, Mark_3, Mark_4, UNet
from trainer import Trainer
from dataset import DIV2K
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

torch.manual_seed(37)
parser = argparse.ArgumentParser(
    description='Train Driver'
)

parser.add_argument('--model', type=str)
parser.add_argument('--n_gpus' , default=1, type=int)
parser.add_argument('--scale_factor' , default=4, type=int)
parser.add_argument('--batch_size' , default=1, type=int)
parser.add_argument('--optimizer' , default='Adam', type=str)
parser.add_argument('--learning_rate' , default=0.001, type=float)
parser.add_argument('--momentum' , default=0.9, type=float)
parser.add_argument('--num_point_samples', default=1000, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--data_dir', type=str)

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]

if __name__ == '__main__':

    rank = 'cuda'

    if args.model == 'Mark_1':
        model = Mark_1().to(rank)
    elif args.model == 'Mark_2':
        model = Mark_2().to(rank)
    elif args.model == 'Mark_3':
        model = Mark_3().to(rank)
    elif args.model == 'Mark_4':
        model = Mark_4().to(rank)
    elif args.model == 'Unet':
        model = UNet().to(rank)


    if args.optimizer == 'SDG':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    loss_fn = torch.nn.L1Loss(reduction='mean')

    train_data = DIV2K(root_dir=args.data_dir, partition='train', downscale_factor=args.scale_factor, num_points=args.num_point_samples)
    val_data = DIV2K(root_dir=args.data_dir, partition='valid', downscale_factor=args.scale_factor, num_points=args.num_point_samples)
    #sub_train = torch.utils.data.Subset(train_data, [1,2,3])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=5)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=5)

    exp_name = '{}_x{}_{}_{}_{}'.format(args.model, args.scale_factor, args.num_point_samples, args.optimizer, args.learning_rate)
    print(exp_name)
    trainer = Trainer(model, rank, train_loader, val_loader, optimizer, scheduler, loss_fn, exp_name)
    trainer.train(args.num_epochs)
