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

arch = {"vgg16":25088,
        "densenet121":1024,
        "alexnet":9216}

def check_gpu(gpu_arg): 
    #If gpu_arg is false then simply return the cpu device 
    if not gpu_arg: 
        return torch.device("cpu") 
   
    #If gpu_arg then make sure to check for CUDA before assigning it 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    if device == "cpu": 
        print("CUDA was not found on device, using CPU instead.") 
    return device 

def initial_model(structure):
    
    if structure == 'vgg16':
        #input_size = 25088
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        #input_size = 1024
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        #input_size = 9216
        model = models.alexnet(pretrained = True)
    else:
        print("sorry {} is not a valid model.".format(structure))
    
    #structure 
    #model = getattr(models, structure)(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False 
    
    return model #,input_size

def model_setup(model, structure, hidden_layer, lr, device):
    
    #for param in model.parameters():
        #param.requires_grad = False 
        
	#Defining model architecture
	
    classifier= nn.Sequential(OrderedDict([('fc1', nn.Linear(arch['structure'], 3800, bias=True)), 
                                   		   ('relu', nn.ReLU()), 
                                      	   ('fc2', nn.Linear(3800,256, bias=True)), 
                                      	   ('relu', nn.ReLU()), 
                                   		   ('fc3', nn.Linear(256,102, bias=True)), 
                                      	   ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier  
    
    if torch.cuda.is_available() and device= gpu:
        model.cuda()
    #model.to(device)
	#Defining necesserily parameters 

	#loss_fn= nn.CrossEntropyLoss()

    criterion= nn.NLLLoss()

    optimizer= optim.Adam(model.classifier.parameters(), lr)
    
    
    
    return model, criterion, optimizer, classifier
    
def network_training(criterion, optimizer, train_loader, valid_loader, model, epochs, device):    

    epochs
    steps=0
    running_loss=0
    print_every=30
    
    model.to(device)
    
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps+=1
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()  
            logps=model.forward(inputs)
            loss= criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss+= loss.item()
   
            if steps % print_every==0:
                valid_loss=0
                accuracy=0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps= model.forward(inputs)
                        batch_loss= criterion(logps, labels)
            
            
                        valid_loss+= batch_loss.item()
           
                        ps= torch.exp(logps)
                        top_p, top_class= ps.topk(1, dim=1)
                        equals= top_class==labels.view(*top_class.shape)
                        accuracy+= torch.mean(equals.type(torch.FloatTensor)).item()
            
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Training Loss: {running_loss/print_every:.3f}.. "
                    f"Validation Loss: {valid_loss/len(valid_loader):.3f}.. "
                    f"Accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()

def network_test(model, criterion, test_loader, gpu): 

    test_loss=0
    accuracy=0
    
    model.to(device)
    
    with torch.no_grad():
        model.eval()
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps= model.forward(inputs)
            batch_loss= criterion(logps, labels)
           
            test_loss+= batch_loss.item()

            ps= torch.exp(logps)
            top_p, top_class= ps.topk(1, dim=1)
            equals= top_class==labels.view(*top_class.shape)
            accuracy+= torch.mean(equals.type(torch.FloatTensor)).item()
        
    print(f"Test Accuracy: {accuracy/len(test_loader):.3f} ")

   