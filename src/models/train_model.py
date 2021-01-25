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
    
net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
