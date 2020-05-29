from __future__ import print_function
#%matplotlib inline
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
import argparse
from IPython.display import HTML
from Model import *



# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")




def common_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', default='data/celeba', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--image_size', default=64, type=int)
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    

    return parser



def train(dataloader, generator, discriminator, optimizer_g, optimizer_d, criterion, num_epochs):
    # Each epoch, we have to go through every data in dataset
    for epoch in range(num_epochs):
        # Each iteration, we will get a batch data for training
        for i, data in enumerate(dataloader, 0):

            # initialize gradient for network
            # send the data into device for computation

            
  
            # Send data to discriminator and calculate the loss and gradient
            # For calculate loss, you need to create label for your data


        
            ## Using Fake data, other steps are the same.
            # Generate a batch fake data by using generator
            
            
            # Send data to discriminator and calculate the loss and gradient
            # For calculate loss, you need to create label for your data

            
            # Update your network

            
            
            # Record your loss every iteration for visualization
            
            
            # Use this function to output training procedure while training
            # You can also use this function to save models and samples after fixed number of iteration
            if i % 50 == 0:
                print(.....)

     
            # Remember to save all things you need after all batches finished!!!
        
        

def main(args):
    # Create the dataset by using ImageFolder(get extra point by using customized dataset)
    # remember to preprocess the image by using functions in pytorch
    dataset = 
    # Create the dataloader
    dataloader = 
    


    # Create the generator and the discriminator()
    # Initialize them 
    # Send them to your device
    generator = 
    discriminator = 

    

    

    # Setup optimizers for both G and D and setup criterion at the same time
    optimizer_g = 
    optimizer_d =
    criterion = 
    
    
    # Start training~~
    
    train(dataloader, generator, discriminator, optimizer_g, optimizer_d, criterion, args.num_epochs)
    


if __name__ == '__main__':
    args = common_arg_parser()
    main(args)