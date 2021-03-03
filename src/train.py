import model
import trainer
from dataset import SRx4Dataset
import argparse
import torch
torch.manual_seed(37)
parser = argparse.ArgumentParser(
    description='Train Driver'
)

parser.add_argument('--model' , default='Mark_1', type=str)
parser.add_argument('--batch_size' , default=32, type=int)
parser.add_argument('--optimizer' , default='SDG', type=str)
parser.add_argument('--learning_rate' , default=0.1, type=float)
parser.add_argument('--momentum' , default=0.9, type=float)
parser.add_argument('--num_point_samples', default=1000, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)


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

if args.optimizer == 'SDG':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=5)

loss_fn = torch.nn.L1Loss(reduction='mean')
#loss_fn = torch.nn.MSELoss(reduction='mean')

data = SRx4Dataset(num_points=args.num_point_samples, transform=True)
num_train = int(0.8*len(data))
num_val = len(data) - num_train
train_data, val_data = torch.utils.data.random_split(data, [num_train, num_val])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True)

exp_name = '{}_{}_{}_{}'.format(args.model, args.num_point_samples, args.optimizer, args.learning_rate)

trainer = trainer.Trainer(model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, exp_name)
trainer.train(args.num_epochs)