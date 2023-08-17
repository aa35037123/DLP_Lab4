import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

# data_range: 0-1 value
def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  # normalize
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        self.kl_anneal_type = args.kl_anneal_type
        self.kl_anneal_cycle = args.kl_anneal_cycle # default is 10, which means how many epochs a period
        self.kl_anneal_ratio = args.kl_anneal_ratio
        self.current_epoch = current_epoch
        self.beta_start = 0
        self.beta_stop = 1
        
        self.beta = self.beta_start
        self.iter_count = 0  
        
    # update per batch
    def update(self):
        self.current_epoch += 1
        if self.kl_anneal_type == 'Monotonic' or self.kl_anneal_type == 'Cyclical':  
        # if self.beta_cycle > 0 and self.current_epoch % self.beta_cycle == 0 :
            self.beta = self.frange_cycle_linear(n_iter=self.current_epoch, start=self.beta_start,
                                                    stop=self.beta_stop, n_cycle=self.kl_anneal_cycle, 
                                                    ratio=self.kl_anneal_ratio)
        # without kl_annealing
        else:
            self.beta = self.beta
            
    def get_beta(self):
        return self.beta

    # return beta value
    # if self.kl_anneal_type == monotonic, then kl_anneal_ratio = 1
    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        
        if self.kl_anneal_type == 'Monotonic':  
        # if self.beta_cycle > 0 and self.current_epoch % self.beta_cycle == 0 :
            iter_with_cycle = (self.current_epoch % n_cycle) 
            return max(start, min(stop, start+(stop-start) / n_cycle * iter_with_cycle)) 
        elif self.kl_anneal_type == 'Cyclical':
            # n_iter means how many iter will be executed in n_cycle
            # calculate iter number in current period 
            # times ratio :
            #   if ratio > 1: the change of beta will faster
            #   if ratio < 1 : the change of beta will slower
            #   if ratio == 1: beta changes uniformly 
            iter_with_cycle = (self.current_epoch % n_cycle) * ratio
            return max(start, min(stop, start+(stop-start) / n_cycle * iter_with_cycle)) 
        

class VAE_Model(nn.Module):
    # the purpose od VAE is generate output frame of next time
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        # teacher forcing rate, higher means there are more chance we need to give model the groundtruth vice versa
        self.tfr = args.tfr 
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        self.path_train_ckpt = os.path.join(self.args.save_root, 'train', 'check_point_wokl')
        self.path_train_fig = os.path.join(self.args.save_root, 'train', 'fig_file')
        self.path_train_loss_compare_fig = os.path.join(self.path_train_fig, 'loss_compare.png')
        self.path_train_tfr_fig = os.path.join(self.path_train_fig, 'tfr_curve.png')
        
        self.path_train_gif = os.path.join(self.args.save_root, 'train', 'gif_file')
        
        self.path_train_csv = os.path.join(self.args.save_root, 'train', 'csv_file')
        self.path_train_loss_cyclical = os.path.join(self.path_train_csv, 'loss_Cyclical.csv')
        self.path_train_loss_monotonic = os.path.join(self.path_train_csv, 'loss_Monotonic.csv')
        self.path_train_loss_wokl = os.path.join(self.path_train_csv, 'loss_Wokl.csv') # not using kl annealing
        
        
        self.path_val_fig = os.path.join(self.args.save_root, 'val', 'fig_file')
        self.path_val_gif = os.path.join(self.args.save_root, 'val', 'gif_file')
        self.path_val_csv = os.path.join(self.args.save_root, 'val', 'csv_file')
        
        
    def forward(self, img, label):
        frame_encoded = self.frame_transformation(img)
        label_encoded = self.label_transformation(label)
        # Concatenate the encoded features
        # combined_features = torch.cat((frame_encoded, label_encoded), dim=1)
        # Predict Gaussian parameters in the latent space
        z, mu, logvar = self.Gaussian_Predictor(frame_encoded, label_encoded)
        decoded = self.Decoder_Fusion(frame_encoded, label_encoded, z)
        generated_output = self.Generator(decoded)
        
        return generated_output 
    
    def training_stage(self): # In training, a batch has 630 frames
        df = pd.DataFrame()
        tfr_list = []
        loss_list = []
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False 
            model_loss = 0.0
            tfr_list.append(self.tfr)
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                # print(f'Shape of origin img : {img.shape}')
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                beta = self.kl_annealing.get_beta() # get weight parameter of KL divergence
                loss, generated_frames, label_list = self.training_one_step(img, label, adapt_TeacherForcing, beta)
                model_loss += loss.detach().cpu()
                for i in range(self.batch_size):
                    self.make_gif(images_list=generated_frames[i], 
                                    img_name=os.path.join(self.path_train_gif, f'ep{self.current_epoch}_pred{i}.gif'))
                    self.make_gif(images_list=label_list[i], 
                                    img_name=os.path.join(self.path_train_gif, f'ep{self.current_epoch}_pose{i}.gif')) 
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {:.1f}'.format(self.tfr, beta), 
                                    pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {:.1f}'.format(self.tfr, beta), 
                                    pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            avg_loss = model_loss / len(train_loader.dataset)
            loss_list.append(avg_loss.detach().item())
            # preserve model checkpoint cyclely
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.path_train_ckpt, f"epoch={self.current_epoch}.ckpt"))
        
            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
        # fig_tfr = self.plot_tfr(tfr_list)
        # fig_tfr.savefig(os.path.join(self.path_train_fig, 'teacher_forcing_rate.png'))
        df[f'train w/{self.args.kl_anneal_type}'] = loss_list
        df.to_csv(os.path.join(self.path_train_csv, f'loss_{self.args.kl_anneal_type}_ep{self.current_epoch-self.args.num_epoch}-{self.current_epoch}.csv'), index=False)    
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        # print(f'Shape of val_loader : {val_loader.shape}')
        best_psnr = float('-inf')
        best_psnr_fig = None
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)): # In val, a batch has 630 frames
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            avg_loss, avg_psnr, generated_frames, label_list, fig = self.val_one_step(img, label)
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr.detach().cpu()
                best_psnr_fig = fig
            
            self.make_gif(images_list=generated_frames[0].detach().cpu(), 
                            img_name=os.path.join(self.path_val_gif, f'ep={self.current_epoch}_valid_frame.gif'))
            self.make_gif(images_list=label_list[0].detach().cpu(), 
                            img_name=os.path.join(self.path_val_gif, f'ep={self.current_epoch}_valid_pose.gif'))
            self.tqdm_bar('val', pbar, avg_loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
        if best_psnr_fig is None:
            print(f'Epoch={self.current_epoch}, avg_psnr : {avg_psnr.detach().cpu()}')
            return
        best_psnr_fig.savefig(os.path.join(self.path_val_fig, f'ep={self.current_epoch}_psnr.png'))
        
    def training_one_step(self, img, label, adapt_TeacherForcing, beta):
        # Assume img is a batch of past frames and label is a batch of label frames
        # img shape: (num_frames=(train_vi_len), batch_size, channels, height, width)
        # label shape: (num_frames=(train_vi_len), batch_size, channels, height, width)
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        # print(f'shape of img : {img.shape}')
        past_frame = img[0]
        current_frame = img[1]
        current_label = label[1]
        generated_frames = [past_frame.detach().cpu()]
        label_list = []
        # kl_loss_list = []
        # mse_loss_list = []
        loss_list = [] # have self.train_vi_len-1 data point, without img[0] which is provided as first input
        for i in range(1, self.train_vi_len):
            current_frame_encoded = self.frame_transformation(current_frame)
            past_frame_encoded = self.frame_transformation(past_frame)
            current_label_encoded = self.label_transformation(current_label)
            z, mu, logvar = self.Gaussian_Predictor(current_frame_encoded, current_label_encoded)            
            # Combine current label and past frame for decoding
            decoded = self.Decoder_Fusion(past_frame_encoded, current_label_encoded, z)
            # Generate the output frame using the generator
            generated_output = self.Generator(decoded)
            generated_frames.append(generated_output.detach().cpu())
            label_list.append(label[i].detach().cpu())
            mse_loss = self.mse_criterion(generated_output, current_frame)
            # mse_loss_list.append(mse_loss)
            
            # Calculate the KL divergence loss
            kl_loss = kl_criterion(mu, logvar, self.batch_size) * beta
            # kl_loss_list.append(kl_loss)
            loss_list.append((mse_loss+kl_loss))
            if adapt_TeacherForcing:
                past_frame = img[i]
            else:
                past_frame = generated_output
            
            current_frame = img[i]
            current_label = label[i] # this means next label of next frame
        
        # Calculate the total loss
        total_loss = sum(loss_list)
        
        # Update the network : backpropagation and optimization
        self.optim.zero_grad() # clear the gradient before backpropagation
        total_loss.backward() # calculate the gradient of neuro network 
        self.optimizer_step() # update the neuro network use the gradient 

        # change to origin after validation
        generated_frames = stack(generated_frames, dim=0).permute(1, 0, 2, 3, 4) # change tensor into (B, seq, C, H, W)
        label_list = stack(label_list, dim=0).permute(1, 0, 2, 3, 4) # change tensor into (B, seq, C, H, W)
        
        return total_loss, generated_frames, label_list
    
    def plot_tfr(self, tfr_list):
        num_epochs = range(1, self.args.num_epoch+1)
        fig = plt.figure(figsize=(10, 6))
        plt.plot(num_epochs, tfr_list, label=f'tfr')
        plt.xlabel('Epoch')
        plt.ylabel('Teacher Forcing Rate')
        plt.legend()
        plt.title('Teacher Forcing Rate - Epoch Curve')
        
        return fig
    
    def plot_psnr(self, psnr_list, avg_psnr):
        psnr_list_cpu = [psnr.cpu().numpy() for psnr in psnr_list] # Move tensors to CPU and convert to numpy arrays
        val_vi_indices = range(2, self.val_vi_len+1) # predict frame is started from second frame
        fig = plt.figure(figsize=(10, 6))
        plt.plot(val_vi_indices, psnr_list_cpu, label=f'Avg_PSNR:{avg_psnr:.3f}')
        plt.xlabel('Frame index')
        plt.ylabel('PSNR')
        plt.legend()
        plt.title('Per Frame Quality (PSNR)')
        
        return fig
        
    def val_one_step(self, img, label):
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        past_frame = img[0]
        generated_frames = [past_frame.detach().cpu()]
        label_list = []
        loss_list = []
        psnr_list = []
        for i in range(1, self.val_vi_len):
            past_frame_encoded = self.frame_transformation(past_frame)
            current_label_encoded = self.label_transformation(label[i])
            # N_dim : noise dimensionm 
            # torch.cuda make z processing on GPU if it's available
            z = torch.cuda.FloatTensor(1, self.args.N_dim, self.args.frame_H, self.args.frame_W).normal_()
            decoded = self.Decoder_Fusion(past_frame_encoded, current_label_encoded, z)
            generated_output = self.Generator(decoded)
            generated_frames.append(generated_output.detach().cpu())
            label_list.append(label[i].detach().cpu())
            loss_list.append(self.mse_criterion(generated_output, img[i]).detach().cpu())
            psnr_list.append(Generate_PSNR(generated_output, img[i]).detach().cpu())
            past_frame = generated_output

        # change to origin after validation
        generated_frames = stack(generated_frames, dim=0).permute(1, 0, 2, 3, 4) # change tensor into (B, seq, C, H, W)
        label_list = stack(label_list, dim=0).permute(1, 0, 2, 3, 4) # change tensor into (B, seq, C, H, W)
        avg_loss = sum(loss_list) / (self.val_vi_len-1)
        avg_psnr = sum(psnr_list) / (self.val_vi_len-1)
        
        fig = self.plot_psnr(psnr_list, avg_psnr)
        
        return avg_loss, avg_psnr, generated_frames, label_list, fig
    
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
        
        # concat all img in new_list into a GIF file            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size, # default batch_size = 2
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        if self.current_epoch >= self.tfr_sde: # self.tfr_sde is the epoch tfr start to decay
            self.tfr *= (1 - self.tfr_d_step)
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path) # load model weight from check_point
            # state_dict is a simple dict, each layer map to one tensor  
            self.load_state_dict(checkpoint['state_dict'], strict=True) # load model state dict into checkpoint 
            self.args.lr = self.args.lr
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            # After 2, 4 epoch, lr = le * gamma
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch'] # ensure that training can go with correct epoch

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.) # clip the gradient, limit gradient not to exceed 1.0  
        self.optim.step()
    def show_result_df(self, df):
        fig = plt.figure(figsize=(10, 6))
        for name in df.columns[1:]:
            plt.plot('epoch', name, data=df)
        plt.legend()
        return fig
    def plot_loss(self):
        df_cyclical = pd.read_csv(self.path_train_loss_cyclical) if os.path.exists(self.path_train_loss_cyclical) else None
        df_monotonic = pd.read_csv(self.path_train_loss_monotonic) if os.path.exists(self.path_train_loss_monotonic) else None
        df_wokl = pd.read_csv(self.path_train_loss_wokl) if os.path.exists(self.path_train_loss_wokl) else None
        if (df_cyclical is None) or (df_monotonic is None) or (df_wokl is None):
            print(f'Not all kl anneaing method are prepared.')
            return
        df_epochs = range(1, self.args.num_epoch)
        df_all_kl_annealing = pd.concat([df_epochs, df_cyclical, df_monotonic, df_wokl], axis=1, ignore_index=False)
        fig = self.show_result_df(df_all_kl_annealing)
        fig.savefig(self.path_train_loss_compare_fig)
"""
--save_root : use in training process, save the model weight
--ckpt_path : use to load pre-train model weight 
"""

def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    print('GPU is available now.') if torch.cuda.is_available() else print('GPU is not available now.')
    print(f'Device Name : {torch.cuda.get_device_name(torch.cuda.current_device())}')
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint() # initial kl_annealing
    if args.test:
        model.eval()
    else:
        model.training_stage()
        
        # plot loss fig of different kl_rate in training 
        model.plot_loss()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    

    

    args = parser.parse_args()
    
    main(args)
