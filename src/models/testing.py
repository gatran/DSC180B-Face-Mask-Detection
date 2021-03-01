from __future__ import print_function
from __future__ import division
import argparse
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
from torchvision.utils import save_image


from training import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--model-name', type=str, default="resnet",
                        help='Model name')
    parser.add_argument('--model-path', type=str, default=os.path.join('../DSC180B-Face-Mask-Detection/models', 'model_resnet_best_val_acc_0.955.pt'),
                        help='Load model path')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    args = get_args()
    # Top level data directory. Here we assume the format of the directory conforms
    # to the ImageFolder structure
    data_dir = "/datasets" + "/MaskedFace-Net"
    # Model path
    model_path = args.model_path
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = args.model_name
    # Number of classes in the dataset
    num_classes = 3
    # Batch size for training (change depending on how much memory you have)
    batch_size = args.batch_size
     
    #New test case
    #Go to the correct file path

    #Accuracy on the validation set with a batch size of 4
    #Load in the saved training model

    #Change filepath back to where the dataset is
    #os.chdir("/datasets" + "/MaskedFace-Net")
    input_size = 0
    if model_name == "resnet":
        input_size = 224
    
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'validation']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'validation']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if args.use_cuda else "cpu")
    
    maskedface_net_val = image_datasets['validation']
    val_sub = torch.utils.data.Subset(maskedface_net_val, np.random.choice(len(maskedface_net_val), 1, replace=False))
    data_loader_val_sub = torch.utils.data.DataLoader(val_sub,
                                                      batch_size=4, #4
                                                      shuffle=True)
    model = torch.load(model_path)
    
    # Send the model to GPU
    model = model.to(device)
    #Accuracy section
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader_val_sub:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the validation set: %d %%' % (100 * correct / total))
    
    classes = ('correctly masked', 'incorrectly masked', 'not masked')
    
    _, predicted = torch.max(outputs, 1)
    results = ' | '.join('%5s' % classes[predicted[j]] for j in range(1))
    print('Model Prediction: ', ' | '.join('%5s' % classes[predicted[j]] for j in range(1)))
    print('Please visit /results/model_prediction for the image')
    save_image(inputs[0].cpu(),'results/model_prediction/' + "img_{0}_prediction_{1}.jpg".format(random.randint(1,len(maskedface_net_val)), results))
    
   