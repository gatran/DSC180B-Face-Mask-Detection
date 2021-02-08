import numpy as np 
import pandas as pd
import torch
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from pathlib import Path
from os.path import dirname, abspath
import seaborn as sns

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 1 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 1, 3)
        self.conv2 = nn.Conv2d(1, 1, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(529, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #print(x.shape)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    
    #Load in the saved training model
    net = Net()
    net.load_state_dict(torch.load("src/models/model_v1.pt"))
    
    #Change directory to where the datasets are
    os.chdir("/datasets" + "/MaskedFace-Net")

    #Read in the test data, grab a subset, and apply transformations on the dataset
    transform_val = transforms.Compose(
        [transforms.Resize((100, 100)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    maskedface_net_val = torchvision.datasets.ImageFolder("../MaskedFace-Net/validation", transform=transform_val)
    val_sub = torch.utils.data.Subset(maskedface_net_val, np.random.choice(len(maskedface_net_val), 10, replace=False))
    data_loader_val_sub = torch.utils.data.DataLoader(val_sub,
                                                      batch_size=32, #4
                                                      shuffle=True)
    
    #Accuracy section
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader_val_sub):
            inputs, labels = data
            outputs = net(inputs)
            predicted = labels
            total += labels.size(0)
            label_tensor_maxs = torch.tensor([torch.argmax(x) for x in labels])
            correct += (predicted == label_tensor_maxs).sum().item()

    print('Accuracy of the model on the test subset: %d %%' % (100 * correct / total))
    
    