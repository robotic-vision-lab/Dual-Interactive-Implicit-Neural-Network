import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Trainer(object):
    def __init__(self, model, device, train_loader, val_loader, optimizer, scheduler, loss_fn, exp_name):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.exp_path = os.path.join('experiments', exp_name)
        if not os.path.exists(self.exp_path):
            print(self.exp_path)
            os.makedirs(self.exp_path)
        else:
            for f in os.listdir(self.exp_path):
                os.remove(os.path.join(self.exp_path, f))
        self.writer = SummaryWriter(self.exp_path)
        self.val_min = None
        self.last_checkpoint=None

    def train_step(self, batch):
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def compute_loss(self, batch):
        lr_img, points, gt = batch
        lr_img = lr_img.to(self.device)
        points = points.to(self.device)
        gt = gt.to(self.device)
        out = self.model(lr_img, points)
        loss = self.loss_fn(out, gt) #(N,num_points,1)
        return loss

    def compute_val_loss(self):
        self.model.eval()
        val_loss = 0
        for batch in self.val_loader:
            val_loss += self.compute_loss(batch).item()
        return val_loss/len(self.val_loader)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            with tqdm(self.train_loader, unit='batch') as tepoch:
                tepoch.set_description(f"Train {epoch}")
                self.model.train()
                train_loss = 0
                for i, batch in enumerate(tepoch):
                    loss = self.train_step(batch)
                    train_loss += loss
                    tepoch.set_postfix(loss=train_loss/(i+1))
                train_loss /= len(self.train_loader)
                self.writer.add_scalar('train_loss/epoch', train_loss, epoch)
                
                with tqdm(self.val_loader, unit='batch') as vepoch:
                    vepoch.set_description(f"Valid {epoch}")
                    self.model.eval()
                    val_loss = 0
                    for i, batch in enumerate(vepoch):
                        loss = self.compute_loss(batch)
                        val_loss += loss.item()
                        vepoch.set_postfix(loss=val_loss/(i+1))
                    val_loss /= len(self.val_loader)
                    if self.val_min is None:
                        self.val_min = val_loss
                        #self.update_checkpoint()
                    if val_loss < self.val_min:
                        self.val_min = val_loss
                        self.update_checkpoint()
                    self.writer.add_scalar('val_loss/epoch', val_loss, epoch)
            self.scheduler.step()
            torch.save(self.model.state_dict(), os.path.join(self.exp_path, 'checkpoint_latest'))

    def update_checkpoint(self):
        if self.last_checkpoint is not None:
            os.remove(self.last_checkpoint)
        path = os.path.join(self.exp_path, 'checkpoint_%.4f'%(self.val_min))
        torch.save(self.model.state_dict(), path)
        self.last_checkpoint = path