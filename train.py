import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import tensor
from torch import optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms 
import PIL
from PIL import Image 
import argparse


import utility
import network

#resnet18 = models.resnet18(pretrained=True)
#alexnet = models.alexnet(pretrained=True)
#vgg16 = models.vgg16(pretrained=True)

#models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}




def init_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./flowers', help='path to the folder of flowers')
    parser.add_argument('--save_dir', dest='save_dir', default='checkpoint.pth', help='path to the saving file')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'densenet121', 'alexnet'], help='path to the CNN Model Architecture')
    parser.add_argument('--input_size', type=int, default=25088, choices=[25088, 1024, 9216], help='input size of CNN Model Architecture')    
    parser.add_argument('--gpu', dest='gpu', default='gpu', help='path to the destination of device')
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.001, help='path to thelearning rate')
    parser.add_argument('--epochs', dest='epochs', type=int, default=1 , help='number of epochs')
    parser.add_argument('--hidden_units', type=int, default=3800, help='number of hidden units' )
    args = parser.parse_args()    
    
    data_dir=args.data_dir
    save_dir=args.save_dir
    structure=args.arch
    input_size= args.input_size
    lr=args.learning_rate
    epochs=args.epochs
    hidden_layer=args.hidden_units
    gpu_arg=args.gpu
    #model = getattr(models, 'structure')(pretrained=True)
    
    
        
    return args

   
def main():

    
    args = init_argparse()
   
    def check_gpu(gpu_arg): 
    #If gpu_arg is false then simply return the cpu device 
        if not gpu_arg: 
            return torch.device("cpu") 
   
    #If gpu_arg then make sure to check for CUDA before assigning it 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        if device == "cpu": 
            print("CUDA was not found on device, using CPU instead.") 
        return device 
    #device = torch.device('cuda' if torch.cuda.is_available() and args.gpu==True else 'cpu')     
   
#load initial model
    model= network.initial_model(args.arch)
    print('load initial model..')

    #load dataset:
    train_data, valid_data, test_data, train_loader, valid_loader, test_loader= utility.pre_data(args.data_dir)
    print('load dataset..')
    
  
    #build model
    model, criterion, optimizer, classifier = network.model_setup(model, args.input_size, args.arch, args.gpu, args.hidden_units, args.learning_rate)
    print('build model arch..')
    
    #train network
    network.network_training( criterion, optimizer, train_loader, valid_loader, model, args.epochs, args.device)
    print('Network Training..')
    
    #test network
    network.network_test(model, criterion, test_loader, args.device)
    print('Test Network..')
    
    #save model
    utility.save_checkpoint(train_data, model, optimizer, args.save_dir)
    print('save model..')
    

if __name__== "__main__":
    main()