import sys
sys.path.append('model')
import time
import os
import random
from tqdm import tqdm
import wandb
import torch
from torch import nn
import numpy as np

from utils.file_utils import get_logger
from dataloader.dsec_full import make_data_loader
from loss import SILogLoss, BinsChamferLoss, compute_metrics
from evaluate import get_vis_prog

import math


####Important####
from model.adabins.unet_adaptive_bins import UnetAdaptiveBins
####Important####

from datetime import datetime
from torchvision.transforms import v2
from flow_vis import flow_to_color

# from evaluate import get_vis_prog

from importlib import import_module

import weakref

MAX_FLOW = 400
SUM_FREQ = 100
VIS_FREQ = 1000
SAVE_FREQ = 10000

CROP_HEIGTH = 288
CROP_WIDTH = 384

# Half precision
#torch.set_default_dtype(torch.float16)

class Loss_Tracker:
    def __init__(self, wandb, sum_frequency = 100):
        self.running_loss = {}
        self.wandb = wandb
        self.sum_frequency = sum_frequency
        self.first_step = True

    def push(self, metrics, step):
        # self.total_steps += 1
        
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0
            
            self.running_loss[key] += metrics[key]

        if self.first_step:
            self.first_step = False
            return
        
        if step % self.sum_frequency == 0:
            if self.wandb:
                for key in self.running_loss:
                    wandb.log({key: self.running_loss[key]/SUM_FREQ}, step = step)
                    
                # wandb.log({'Segmentation Crossentropy':self.running_loss['seg_loss']/SUM_FREQ}, step=self.total_steps)
                # wandb.log({'Experimental Loss':self.running_loss['experimental_loss']/SUM_FREQ}, step=self.total_steps)
            self.running_loss = {}
        
    def state_dict(self):
        return {"running_loss":self.running_loss,
                "wandb":self.wandb}
    
    def load_state_dict(self, state_dict:dict):
        keys = ["running_loss", "wandb"]
        for key in keys:
            self.__setattr__(key, state_dict[key])
            

class Trainer:
    def __init__(self, args):
        self.args = args

        self.model = UnetAdaptiveBins.build(100)
        self.model = self.model.cuda()
        # self.model = nn.DataParallel(self.model, device_ids=[0])

        self.date_label = datetime.now().strftime("%Y-%m-%d")
        self.save_path = os.path.join(args.checkpoint_dir, self.date_label)

        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

        #Loader
        self.train_loader = make_data_loader('trainval', args.batch_size, args.num_workers, strategy='continuous')
        print('train_loader done!')

        self.current_step = 0
        self.total_steps = args.num_steps
        self.lr_max = args.lr
        self.lr_start = args.lr / 25
        self.lr_end = args.lr / 75 
        self.warmup_ratio = 0.3

        #Optimizer and Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=1e-2
        )

        lr_lambda = lambda step : self.get_lr_by_step(weakref.ref(self), step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        #Loss
        self.w_chamfer = args.w_chamfer
        self.criterion_ueff = SILogLoss()
        self.criterion_bins = BinsChamferLoss()

        #Logger
        self.checkpoint_dir = args.checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.writer = get_logger(os.path.join(self.checkpoint_dir, self.date_label, 'train.log'))
        self.tracker = Loss_Tracker(args.wandb, sum_frequency = SUM_FREQ)

        #Loading checkpoint
        self.continue_training = args.continue_training
        self.old_ckpt_path = args.model_path
        self.previous_step = None

        if self.old_ckpt_path == "":
            if self.continue_training:
                print("Cannot continue training without a pretrained model checkpoint. Please provide '--model_path'")
                self.continue_training = False
        else:
            if os.path.isfile(self.old_ckpt_path):
                params_ckpt_path = os.path.join(os.path.dirname(self.old_ckpt_path), "params_checkpoint")
                params_ckpt_path = params_ckpt_path if self.continue_training else None
                self.previous_step = self.load_ckpt(self.old_ckpt_path, params_ckpt_path)
                self.writer.info(f"Loaded the checkpoint at '{self.old_ckpt_path}'.")
                if self.continue_training:
                    self.writer.info("Loaded parameters for continuous learning.")
            else:
                print("Couldn't find a checkpoint file at '{}'".format(self.old_ckpt_path))

        self.writer.info('====A NEW TRAINING PROCESS====')

    @staticmethod
    def get_lr_by_step(weakself, current_step):
         # Linear warmup
        self = weakself()
        warmup_steps = int(self.total_steps * self.warmup_ratio)
        if current_step < warmup_steps:
            return self.lr_start / self.max_lr + (current_step / warmup_steps) * (1 - self.lr_start / self.max_lr)
        # Cosine annealing after warmup
        progress = (current_step - warmup_steps) / (self.total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return cosine_decay * (1 - self.lr_end / self.max_lr) + self.lr_end / self.max_lr

    def train(self):
        # self.writer.info(self.model)
        self.writer.info(self.args)
        self.model.train()
        
        current_step = 0
        vis_steps = 0
        if self.continue_training:
            current_step = self.previous_step
            vis_steps = int(current_step / VIS_FREQ)

        keep_training = True
        while keep_training:

            bar = tqdm(self.train_loader,total=len(self.train_loader), ncols=60)
            for voxel, depth_gt, depth_valid, _ in bar:
                # voxel1, voxel2, flow_map, valid2D = self.apply_transforms(data_items)
                self.optimizer.zero_grad()

                bin_edges, pred = self.model(voxel.cuda())

                # TODO : Implement loss function and use it here
                l_dense = self.criterion_ueff(pred, depth_gt.cuda(), depth_valid.cuda())
                l_chamfer = self.criterion_bins(bin_edges, pred)
                
                loss = l_dense + self.w_chamfer * l_chamfer

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
                self.scheduler.step()

                loss_metrics = {}
                loss_metrics['Loss/SILL'] = l_dense.item()
                loss_metrics['Loss/Chamfer'] = l_chamfer.item()

                loss_metrics['Metrics/abs_rel'] = torch.mean(torch.abs(depth_gt - pred) / depth_gt).item()
                loss_metrics['Metrics/sq_rel'] = torch.mean((depth_gt - pred) ** 2 / depth_gt).item()
                loss_metrics['Metrics/rmse'] = torch.sqrt(torch.mean((depth_gt - pred) ** 2)).item()
                
                self.tracker.push(loss_metrics, step=current_step)
                
                bar.set_description(f'Step: {current_step}/{self.args.num_steps}')
                current_step +=1
                self.current_step = current_step

                if current_step and current_step % VIS_FREQ == 0 and self.args.wandb:
                    self.model.eval()
                    vis_steps += 1
                    with torch.no_grad():
                        # Get progress visualizations
                        visualizations = get_vis_prog(self.model)
                        for i, vis in visualizations:
                            wandb.log({'progress_{}'.format(i+1): wandb.Image(vis)})

                    self.model.train()

                if current_step and current_step % SAVE_FREQ == 0:
                    # Checkpoint savepath
                    ckpt = os.path.join(self.save_path, f'checkpoint_{current_step}')

                    # Save checkpoint with parameters for continuous training
                    params_state = {"step":current_step,
                            "model":self.model.state_dict(),
                            "optimizer":self.optimizer.state_dict(),
                            "scheduler":self.scheduler.state_dict(),
                            "loss_tracker":self.tracker.state_dict()}

                    torch.save(params_state, ckpt)

                if current_step > self.args.num_steps:
                    keep_training = False
                    break
            
            time.sleep(0.03)
        
        # Save final Checkpoint
        model_ckpt_path = os.path.join(self.save_path, "checkpoint.pth")
        torch.save(self.model.state_dict(), model_ckpt_path)

        return model_ckpt_path
    
    def load_ckpt(self, ckpt_path:str, continuous = False):
        if os.path.isfile(ckpt_path):
            # Load the model
            checkpoint = torch.load(ckpt_path)
            if "model" in checkpoint.keys():
                self.model.load_state_dict(checkpoint["model"], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)

            # Load training params
            if continuous:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                    self.scheduler.load_state_dict(checkpoint["scheduler"])
                    self.tracker.load_state_dict(checkpoint["loss_tracker"])
                except KeyError:
                    print("It seems like one or more parameters are missing in the checkpoint. Cannot continue learning.")
                    self.continue_training = False
                    return
                
                return checkpoint["step"]
        else:
            print("Warning : No checkpoint was found at '{}'".format(ckpt_path))

    # def apply_transforms(self, data_items):
    #     transformed_items = data_items
    #     if self.crop:
    #         crop_size = (CROP_HEIGTH, CROP_WIDTH)
    #         rand_crop = v2.RandomCrop(crop_size)
    #         i, j, h, w = rand_crop.get_params(data_items[0], output_size = crop_size)

    #         transformed_items = [v2.functional.crop(item, i, j, h, w) for item in transformed_items]

    #     if self.hflip and torch.rand() > 0.5:
    #         transformed_items = [v2.functional.hflip(item) for item in transformed_items]

    #     return transformed_items
      

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


        
if __name__=='__main__':
    import argparse


    parser = argparse.ArgumentParser(description='TMA')
    #training setting
    parser.add_argument('--num_steps', type=int, default=200000)
    parser.add_argument('--checkpoint_dir', type=str, default='')
    parser.add_argument('--lr', type=float, default=3.5e-4)

    #dataloader setting
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=8)

    #model setting
    parser.add_argument('--grad_clip', type=float, default=1)

    # loss setting
    parser.add_argument('--w_chamfer', type=float, default=0.8, help="Bins chamfer loss weight")

    #wandb setting
    parser.add_argument('--wandb', action='store_true', default=False)

    #Loading pretrained models
    parser.add_argument('--model_path', type=str, default="", help="Path to existing model to be loaded")
    parser.add_argument('--continue_training', action='store_true', default=False, help="Continue learning with previous params")
        
    args = parser.parse_args()
    set_seed(1)
    if args.wandb:
        wandb_name = args.checkpoint_dir.split('/')[-1]
        wandb.init(name=wandb_name, project='EVDPTH')

    trainer = Trainer(args)
    trainer.train()
    
