import numpy as np 
import pandas as pd
import torch
import os
import argparse
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from pathlib import Path
from os.path import dirname, abspath
import seaborn as sns

from nn_model import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch Size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning Rate')
    parser.add_argument('--num-epochs', type=int, default=2,
                        help='Number of Epochs')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

if __name__ == '__main__':
    #os.chdir("/datasets" + "/MaskedFace-Net")

    args = get_args()
    
    transform_train = transforms.Compose(
        [transforms.Resize((100, 100)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_val = transforms.Compose(
        [transforms.Resize((100, 100)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    maskedface_net_train = torchvision.datasets.ImageFolder("/datasets/MaskedFace-Net/train", transform=transform_train)
    maskedface_net_val = torchvision.datasets.ImageFolder("/datasets/MaskedFace-Net/validation", transform=transform_val)
    maskedface_net_test = torchvision.datasets.ImageFolder("/datasets/MaskedFace-Net/holdout", transform=transform_val)

    data_loader_train = torch.utils.data.DataLoader(maskedface_net_train,
                                              batch_size=args.batch_size, #4
                                              shuffle=True)

    data_loader_val = torch.utils.data.DataLoader(maskedface_net_val,
                                                 batch_size=args.batch_size, #4
                                                 shuffle=True)

    data_loader_test = torch.utils.data.DataLoader(maskedface_net_test,
                                                 batch_size=args.batch_size, #4
                                                 shuffle=True)


        
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9)

    batch_g = []
    ep = []
    for epoch in range(args.num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(data_loader_train):
            if i == 5:
                break
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 293 == 292:    # print every 293 mini batches [37500/(batch size == 32)/4 (for graphical purposes)]
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 293))
                batch_g.append(epoch + 1)
                ep.append(running_loss / 293)
                running_loss = 0.0

    print('Finished Training')

    """
    get_ipython().run_cell_magic('time', '', "#Takes a while, maybe few hours??\nbatch_g = []\nep = []\nfor epoch in range(3):  # loop over the dataset multiple times\n\n    running_loss = 0.0\n    for i, data in enumerate(data_loader_train):\n        # get the inputs; data is a list of [inputs, labels]\n        inputs, labels = data\n        \n        # zero the parameter gradients\n        optimizer.zero_grad()\n\n        # forward + backward + optimize\n        outputs = net(inputs)\n            \n        loss = criterion(outputs, labels)\n        loss.backward()\n        optimizer.step()\n\n        # print statistics\n        running_loss += loss.item()\n        if i % 293 == 292:    # print every 293 mini batches [37500/(batch size == 32)/4 (for graphical purposes)]\n            print('[%d, %5d] loss: %.3f' %\n                  (epoch + 1, i + 1, running_loss / 293))\n            batch_g.append(epoch + 1)\n            ep.append(running_loss / 293)\n            running_loss = 0.0\n\nprint('Finished Training')")
    """

    df = pd.DataFrame(list(zip(batch_g, ep)), columns = ['Epoch', 'Loss'])
    print(df)

    sns.barplot(data=df, x = "Epoch", y = "Loss").set_title("Loss found when averaging on every 4th Loss recorded per Epoch")

    print("The black bar represents the Error or in other words the variability of data")


    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader_val):
            inputs, labels = data
            outputs = net(inputs)
            predicted = labels
            total += labels.size(0)
            label_tensor_maxs = torch.tensor([torch.argmax(x) for x in outputs]) #Used to be labels
            correct += (predicted == label_tensor_maxs).sum().item()

    print('Accuracy of the network on the validation set: %d %%' % (100 * correct / total))
