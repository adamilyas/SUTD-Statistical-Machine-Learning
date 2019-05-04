# sorry kind sir, forgot to import 'os'
import os
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #initialize the layers 
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=16, 
                kernel_size=5, 
                stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, 
                kernel_size=5, 
                stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc_elu_fc = nn.Sequential(
            nn.Linear(7*7*32, 4096),
            nn.ELU(),
            nn.Linear(4096, 10)
        )
        
    def forward(self, x):
        # invoke the layers
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc_elu_fc(out)
    
        return out

def train(model,device,train_loader,optimizer,epoch):
    model.train()
    for batch_idx, (data,target) in enumerate(train_loader):
        data,target = data.to(device),target.to(device)
            
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), 0.001)
        
        # Foward pass
        outputs = model(data)
        loss = criterion(outputs,target)
        
        #Optimizer's step() function is used to update the weights after 
        # backpropogating the gradients
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Epoch:',epoch,',loss:',loss.item())
                     
# defind the variable "pred" which predicts the output and update the variable "correct" 
# to keep track of the no. of correctly classified objects to compute the accuracy of the CNN            
def test(model,device,test_loader, plot=False):
    model.eval()
    correct = 0
    exampleSet = False
    example_data = numpy.zeros([10,28,28])
    example_pred = numpy.zeros(10)
    
    with torch.no_grad():
        for data, target in test_loader:
            data,target = data.to(device), target.to(device)
            
            outputs = model(data)
            _, pred = torch.max(outputs.data,1)
            correct += (pred == target).sum()
            
            if not exampleSet:
                for i in range(10):
                    example_data[i] = data[i][0].to('cpu').numpy()
                    example_pred[i] = pred[i].to('cpu').numpy()
                exampleSet = True
    
    set_accuracy = (100*correct/len(test_loader.dataset)).item()
    print(f'Test set accuracy: {set_accuracy}%')

    if plot:
        fig = plt.figure(figsize=(12,6));
        for i in range(10):
            plt.subplot(2,5,i+1)
            plt.imshow(example_data[i],cmap='gray',interpolation='none')
            plt.title(labels[example_pred[i]])
            plt.axis('off')
        plt.show()

labels = {
    0: 'T-shirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress', 
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Angle boot'
}
    
def xavier_init(m):
    """
    model.apply(xavier_init)
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        print(classname)
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.Linear:
        print(classname)
        nn.init.xavier_uniform_(m.weight)
    
def main():
    NUM_EPOCHS = 10
    LRATE = 0.015
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    # data
    data_path = './data'
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    train_dataset = datasets.FashionMNIST(data_path, train=True, 
                                          download=True, transform=transforms.ToTensor())
    test_dataset = datasets.FashionMNIST(data_path, train=False, 
                                          download=True, transform=transforms.ToTensor())

    batch_size = 100
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True);
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False);
    model = CNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=LRATE)

    # initialize weights using xavier initialize
    model.apply(xavier_init) 
    for epoch in range(1,NUM_EPOCHS + 1):
        test(model,device,test_loader, plot=True)
        train(model,device,train_loader,optimizer,epoch)
    
if __name__ == "__main__":
    main()
