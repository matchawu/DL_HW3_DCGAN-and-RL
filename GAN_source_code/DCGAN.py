# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 03:12:02 2020

@author: wwj
"""


from __future__ import print_function
#%matplotlib inline
# from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
# signal(SIGPIPE, SIG_IGN)
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from Model import *
from datetime import datetime
from torchvision.utils import save_image
import pickle

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
# print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# check time & folder
time = datetime.now().strftime("%d_%H%M")
# print("Now time: ", time)
save_dir = './DCGAN_at_'+time+'/'
# print("[Info] Model will be saved in "+save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    # print('[Info] Make directory '+save_dir+' !')

# check device
# device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")
device = "cpu"
# print("[Info] Using device: ", device)

workers = 2 # for dataloader

def common_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', default='../../img_align_celeba/', type=str)
    parser.add_argument('--batchsize', default=128, type=int)
    parser.add_argument('--img_size', default=64, type=int)
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    args = parser.parse_args()
    
    return args

def save_loss(D_loss, G_loss):
    with open(save_dir+'D_loss.pkl', 'wb') as b:
        pickle.dump(D_loss,b)
    with open(save_dir+'G_loss.pkl', 'wb') as b:
        pickle.dump(G_loss,b)
    print('[Info] Saving D&G losses finished.')
    
    
def save_model(g,d):
        torch.save(g.state_dict(), save_dir+'/G_model.pth')
        torch.save(d.state_dict(), save_dir+'/D_model.pth')
        print('[Info] Saving D&G models finished.')


def train(dataloader, G, D, optimizer_g, optimizer_d, criterion, num_epochs):
    # Each epoch, we have to go through every data in dataset
    # Lists to keep track of progress
    img_list = []
    G_loss = []
    D_loss = []
    iters = 0
    for epoch in range(num_epochs):
        # Each iteration, we will get a batch data for training
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            
            # initialize gradient for network
            # send the data into device for computation
            D.zero_grad()
            
            real_data = data[0].to(device)
            b_size = real_data.size(0)
            
            # labelReal = torch.full((b_size,), 1, device=device, dtype=torch.float) # real hard label
            # labelFake = torch.full((b_size,), 0, device=device, dtype=torch.float) # fake hard label
            
            labelReal = ((1.2 - 1.0) * torch.rand(b_size,) + 1.0).to(device) # real soft label
            labelFake = ((0.3 - 0.0) * torch.rand(b_size,) + 0.0).to(device) # fake soft label
            
            
            ## REAL batch
            # pass real batch through Discriminator
            d_output = D(real_data).view(-1)
            # calculate loss on real batch 
            d_error_real = criterion(d_output, labelReal)
            # calculate gradients
            d_error_real.backward()
            D_x = d_output.mean().item()
            
            
            ## FAKE batch
            # generate noise of latent size(nz=100)
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_data = G(noise)
            # pass fake batch through Discriminator
            d_output = D(fake_data.detach()).view(-1)
            # calculate loss on fake batch 
            d_error_fake = criterion(d_output, labelFake)
            # calculate gradients
            d_error_fake.backward()
            D_G_z1 = d_output.mean().item()

            
            # Update your network
            # gradients from the all-real and all-fake batches
            d_error = d_error_real + d_error_fake
            #  update discriminator
            optimizer_d.step()
            
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            G.zero_grad()
            d_output = D(fake_data).view(-1)
            g_error = criterion(d_output, labelReal)
            g_error.backward()

            D_G_z2 = d_output.mean().item()
            optimizer_g.step()
            
            # Record your loss every iteration for visualization
            # saving G and D's training loss
            G_loss.append(g_error.item())
            D_loss.append(d_error.item())
            
            
            # Use this function to output training procedure while training
            # You can also use this function to save models and samples after fixed number of iteration
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         d_error.item(), g_error.item(), D_x, D_G_z1, D_G_z2))
            
            
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 1000 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake_imgs = G(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake_imgs, padding=2, normalize=True))

                # Grab a batch of real images from the dataloader
                real_batch = next(iter(dataloader))

                # Plot the real images
                plt.figure(figsize=(15,15))
                plt.subplot(1,2,1)
                plt.axis("off")
                plt.title("Real Images")
                plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

                # Plot the fake images from the last epoch
                plt.subplot(1,2,2)
                plt.axis("off")
                plt.title("Fake Images")
                plt.imshow(np.transpose(img_list[-1],(1,2,0)))
                plt.savefig(save_dir+'sample_images_iter'+str(iters)+'.png')
                print('[Info] Save sample image finished.')
            
            iters += 1

     
        # Remember to save all things you need after all batches finished!!!
        if epoch == num_epochs-1:
            # save the losses and models
            save_loss(D_loss, G_loss)
            save_model(G,D)
            # plot the result
            plt.figure(figsize=(10,5))
            plt.title("Generator and Discriminator Loss During Training")
            plt.plot(G_loss,label="G")
            plt.plot(D_loss,label="D")
            plt.xlabel("iterations")
            plt.ylabel("loss")
            plt.legend(loc='upper right')
            plt.savefig(save_dir+'loss.png')
            plt.show()
        
# custom weights initialization called on netG and netD
def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)      

def main(args):
    # Create the dataset by using ImageFolder(get extra point by using customized dataset)
    # remember to preprocess the image by using functions in pytorch
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    resize_size = args.img_size+45 
    transform=transforms.Compose([transforms.Resize((resize_size,resize_size)),
                                  transforms.CenterCrop(args.img_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean,std)])

    dataset = dset.ImageFolder(root=args.dataroot, transform=transform)
    
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize,
                                         shuffle=True, num_workers=workers)
    
     # Plot some training images
    # real_batch = next(iter(dataloader))
    # plt.figure(figsize=(8,8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    # plt.savefig(save_dir+'training_images.png')
    
    
     
    # Create the generator and the discriminator()
    # Initialize them 
    # Send them to your device
    G = Generator().to(device)
    G.apply(weights_init)
    
    D = Discriminator().to(device)
    D.apply(weights_init)

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    

    # Setup optimizers for both G and D and setup criterion at the same time
    optimizer_g = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    
    
    # Start training~~
    
    train(dataloader, G, D, optimizer_g, optimizer_d, criterion, args.num_epochs)
    


if __name__ == '__main__':
    args = common_arg_parser()
    main(args)