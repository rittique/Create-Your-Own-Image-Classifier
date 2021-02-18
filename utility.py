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
import json


def pre_data(data_dir):
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    data_dir = "./flowers"
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
	# TODO: Define your transforms for the training, validation, and testing sets
	#data_transforms  
    train_transforms = transforms.Compose([transforms.RandomRotation(30), 
    	                                   transforms.RandomResizedCrop(224), 
        	                               transforms.RandomHorizontalFlip(), 
            	                           transforms.ToTensor(), 
                	                       transforms.Normalize([0.485, 0.456, 0.406],
                    	                                        [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(225), 
    	                                   transforms.CenterCrop(224), 
        	                               transforms.ToTensor(), 
            	                           transforms.Normalize([0.485, 0.456, 0.406],
                	                                            [0.229, 0.224, 0.225])])



	# TODO: Load the datasets with ImageFolder
	#image_datasets
    train_data = datasets.ImageFolder(data_dir + '/train' , transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid' , transform=test_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test' , transform=test_transforms)

	
	# TODO: Using the image datasets and the trainforms, define the dataloaders
	#dataloaders 
    train_loader= torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader= torch.utils.data.DataLoader(valid_data, batch_size=64)
    test_loader= torch.utils.data.DataLoader(test_data, batch_size=64)
	
    return train_data, valid_data, test_data, train_loader, valid_loader, test_loader
    
    
	
def save_checkpoint(train_data, model, save_dir, optimizer): #classifier, optimizer, epochs, lr, structure, hidden_layer):
    
    #save_dir='checkpoint.pth'
    
    model.class_to_idx = train_data.class_to_idx

    torch.save = ({'model': structure,
                  'input_size': input_size,
                  'output_size': 102,
                  'classifier': model.classifier, 
                  'epoch': epochs,
                  'learning_rate': lr,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'optimizer': optimizer.state_dict()}, 'checkpoint.pth')
    
    checkpoint = {'architecture': model.structure,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, 'checkpoint.pth')
    return checkpoint
   
    
    print('checkpoint saved')
    
    
def load_checkpoint(save_dir):
    
    
    checkpoint= torch.load('save_dir')
    lr=checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    structure = checkpoint['structure']

    
    for param in model.parameters():
        param.requires_grad=False
        
    model.class_to_idx = checkpoint['class_to_idx']
    
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
 

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img= Image.open(image)
    img_transform = transforms.Compose([transforms.Resize(256), 
     	                               transforms.CenterCrop(224), 
        	                            transforms.ToTensor(), 
            	                        transforms.Normalize([0.485, 0.456, 0.406],
                	                                         [0.229, 0.224, 0.225])])
    
    
    image= img_transform(img)
    return image
    
    
def imshow(image, ax=None, title=None):

    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
    
def predict(img_path, model, topk):

    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
   
    
    model.device
    
    
    model_.eval()
    
    
    img = Image.open(img_path)
    imgN = process_image(img_path)
    imgN = imgN.unsqueeze(0)
    imgN = imgN.float()
    
    
    
    with torch.no_grad():
        output = model.forward(imgN.power())
    
       
    f_probability = F.softmax(output.data,dim=1)
    
    
    top_f_probability, indices = torch.topk(f_probability, dim=1, k=topk)
    
    
    #Find the class using the indices
    indices = np.array(indices) 
    index_to_class = {val: key for key, val in model.class_to_idx.items()} 
    top_f_classes = [index_to_class[each] for each in indices[0]]
    
    #Map the class name with collected top-k classes
    f_names = []
    
    for classes in top_f_classes:
            f_names.append(cat_to_name[str(classes)])
            
    return top_f_probability.power().numpy(), f_names


def probability(top_f_probability, top_f_classes): 
    top_f_probability, top_f_classes= predict(img_path, model)
    print(top_f_probability)
    print(top_f_classes)