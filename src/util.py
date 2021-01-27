import numpy as np 
import pandas as pd
import torch
import os
import shutil
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from pathlib import Path
from os.path import dirname, abspath


os.chdir("/datasets" + "/MaskedFace-Net")

def load_dataset(batch_size):
    
    # Changing directory to where the dataset is stored
    os.chdir("/datasets" + "/MaskedFace-Net")
    
    # Transform pipeline for training data
    transform_train = transforms.Compose(
    [transforms.Resize((100, 100)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Transform pipeline for validation data and test data
    transform_val = transforms.Compose(
    [transforms.Resize((100, 100)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    maskedface_net_train = torchvision.datasets.ImageFolder("../MaskedFace-Net/train", transform=transform_train)
    maskedface_net_val = torchvision.datasets.ImageFolder("../MaskedFace-Net/validation", transform=transform_val)
    maskedface_net_test = torchvision.datasets.ImageFolder("../MaskedFace-Net/holdout", transform=transform_val)
    
    return maskedface_net_train, maskedface_net_val, maskedface_net_test

def save_model_performance(stats, save_path)

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        
    name = ["train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    data = dict(zip(name, stats))
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(save_path, "results.csv"), index = False)
    
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model


