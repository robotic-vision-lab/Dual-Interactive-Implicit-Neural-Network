import os
import model
import trainer
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

parser.add_argument('--model' , default='Mark_1', type=str)
parser.add_argument('--scale_factor' , default=4, type=int)
parser.add_argument('--batch_size' , default=None, type=int)
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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '142857'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo(rank, world_size):
    setup(rank, world_size)

    if args.model == 'Mark_1':
        model = model.Mark_1().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    if args.optimizer == 'SDG':
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=5)

    loss_fn = torch.nn.L1Loss(reduction='mean')

    data = DIV2K(root_dir=args.data_dir, partition='train', downscale_factor=args.scale_factor, num_points=args.num_point_samples)
    num_train = int(0.8*len(data))
    num_val = len(data) - num_train
    train_data, val_data = torch.utils.data.random_split(data, [num_train, num_val])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True)

    exp_name = '{}_x{}_{}_{}_{}'.format(args.model, args.scale_factor, args.num_point_samples, args.optimizer, args.learning_rate)
    print(exp_name)
    trainer = trainer.Trainer(model, device, train_loader, val_loader, optimizer, scheduler, loss_fn, exp_name)
    trainer.train(args.num_epochs)

    cleanup()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    if n_gpus < 4:
        print(f"Requires at least 4 GPUs to run, but got {n_gpus}.")
    else:
        run_demo(demo, 4)