import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class Trainer(object):
    def __init__(self, model, loader, optimizer, loss_fn, device, exp_name):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loader = loader
        self.device = device
        self.exp_path = os.path.join('experiments', exp_name) 
        if not os.path.exists(self.exp_path):
            print(self.exp_path)
            os.makedirs(self.exp_path)
        self.writer = SummaryWriter(self.exp_path + 'summary_{}'.format(exp_name))
        self.val_min = None

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def compute_loss(self, batch):
        lr_img, points, gt = batch
        out = self.model(lr_img, points)
        loss = loss_fn(out, gt)
        return loss

    def compute_val_loss(self):
        return 0

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            batch_loss = 0
            for batch in self.loader:
                loss = train_step(batch)
                batch_loss += loss 
                self.writer.add_scalar('train_loss/batch', loss, epoch+1/len(self.loader))

            val_loss = self.compute_val_loss()

            if self.val_min is None:
                self.val_min = val_min
            elif val_loss < self.val_min:
                self.val_min = val_loss

            