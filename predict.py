import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import optim
import torchvision
from torchvision import datasets, models, transforms 
from PIL import Image 
import json
import argparse

import utility
import network


def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', default='./flowers/test/2/image_05109.jpg', nargs='*', help='path to image to be classified')
    parser.add_argument('--save_dir', default='checkpoint.pth', nargs='*', help='path to stored model')
    parser.add_argument('--top_k', type=int, default=5, dest='top_k', help='top probablity')
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json', help='file which maps classes to names')
    parser.add_argument('--gpu', dest='gpu', default='gpu', help='path to the destination of device')
    args=parser.parse_args()

    arch = args.arch
    input_file=args.input
    save_dir=args.save_dir
    topk=args.top_k
    flower_names=args.category_names
    
  
    
    return args


#if args.gpu:

 #       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  #  else: 
   #     device = "cpu"
        
def check_gpu(args_gpu):
    if gpu_arg:
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device 
    model.to(device)   

def main():
    
    args = init_argparse()
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    model.to(gpu)
    
    model= utility.load_checkpoint('save_dir')
    
    utiltiy.process_image('input_file')
    
    top_f_probability.power().numpy(), f_names=utility.predict(process_image, model, 'topk', 'device', 'flower_names')
    
    utility.probability(top_f_probability, top_f_classes)
          
    #names= [cat_to_name[str(index + 1)] for index in np.array(probs[1][0])]
    #prob = np.array(probs[0][0])
    
    #i=0
    #while i < number_of_outputs:
     #   print("{} with a prob of {}".format(names[i], prob[i]))
      #  i += 1
        
    #print('DONE')
    
if __name__== "__main__":
    main()