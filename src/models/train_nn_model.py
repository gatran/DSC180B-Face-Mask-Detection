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
    os.chdir("/datasets" + "/MaskedFace-Net")


    transform_train = transforms.Compose(
        [transforms.Resize((100, 100)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_val = transforms.Compose(
        [transforms.Resize((100, 100)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    maskedface_net_train = torchvision.datasets.ImageFolder("../MaskedFace-Net/train", transform=transform_train)
    maskedface_net_val = torchvision.datasets.ImageFolder("../MaskedFace-Net/validation", transform=transform_val)
    maskedface_net_test = torchvision.datasets.ImageFolder("../MaskedFace-Net/holdout", transform=transform_val)

    data_loader_train = torch.utils.data.DataLoader(maskedface_net_train,
                                              batch_size=32, #4
                                              shuffle=True)

    data_loader_val = torch.utils.data.DataLoader(maskedface_net_val,
                                                 batch_size=32, #4
                                                 shuffle=True)

    data_loader_test = torch.utils.data.DataLoader(maskedface_net_test,
                                                 batch_size=32, #4
                                                 shuffle=True)


        
    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    get_ipython().run_cell_magic('time', '', "#Takes a while, maybe few hours??\nbatch_g = []\nep = []\nfor epoch in range(3):  # loop over the dataset multiple times\n\n    running_loss = 0.0\n    for i, data in enumerate(data_loader_train):\n        # get the inputs; data is a list of [inputs, labels]\n        inputs, labels = data\n        \n        # zero the parameter gradients\n        optimizer.zero_grad()\n\n        # forward + backward + optimize\n        outputs = net(inputs)\n            \n        loss = criterion(outputs, labels)\n        loss.backward()\n        optimizer.step()\n\n        # print statistics\n        running_loss += loss.item()\n        if i % 293 == 292:    # print every 293 mini batches [37500/(batch size == 32)/4 (for graphical purposes)]\n            print('[%d, %5d] loss: %.3f' %\n                  (epoch + 1, i + 1, running_loss / 293))\n            batch_g.append(epoch + 1)\n            ep.append(running_loss / 293)\n            running_loss = 0.0\n\nprint('Finished Training')")


    df = pd.DataFrame(list(zip(batch_g, ep)), columns = ['Epoch', 'Loss'])


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

