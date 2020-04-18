#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
import os
import time

start_time = time.time()

# training args
parser = argparse.ArgumentParser(description="Pytorch mnist example for knowledge distillation")
parser.add_argument('--batch-size',type=int, default=128,
                   help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=1000,
                   help='input batch size for testing(default is 1000)')
parser.add_argument('--epochs', type=int, default=10,
                   help='number of the epoches(default is 10)')
parser.add_argument('--lr', type=int, default=0.01,
                   help='learning rate(default: 0.01)')
parser.add_argument('--momentum', type=int, default=0.9,
                   help='SGD momentum (default:0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                   help='disable cuda training')
parser.add_argument('--seed', type=int, default=1,
                   help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                   help='how many batches to wait before logging train status')
parser.add_argument('save_model', action='store_true', default=True,
                   help='saving  current model')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
# prepare random seed
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers':1 ,'pin_memory': True } if use_cuda else {}

# train loader
train_loader = torch.utils.data.DataLoader(
   datasets.MNIST('./data_mnist',train=True, download=True,
                 transform=transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize((0.1307,),(0.3081,))
               ])),
   batch_size=args.batch_size, shuffle=True, **kwargs)

# test loader

test_loader = torch.utils.data.DataLoader(
   datasets.MNIST('./data_mnist',train=False,download=False,
                 transform=transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize((0.1307,),(0.3081,))
               ])),
   batch_size=args.test_batch_size,shuffle=True, **kwargs)

class StudentNet(nn.Module):
   # data shape  (it don't include batch_size, because batch_size does not involved in calculations):
   # B : batch_size
   # data shape (B, channel, hight, width)
   # (B,1,28,28) ->self.conv1-> (B,10,26,26) -> max_pool2d -> (B,10,13,23) 
   #   -> self.conv2 -> (B,30,11,11) -> max_pool2d ->(B,30,5,5) -> self.fc1 -> (B,300) ->self.fc2 -> (B, 10)
   def __init__(self):
       super(StudentNet,self).__init__()
       self.conv1 = torch.nn.Conv2d(1,10,3,1) # Conv2d(input_channel, output_channel, kernel_size, stride)
       self.conv2 = torch.nn.Conv2d(10,30,3,1)
       self.fc1 = torch.nn.Linear(5*5*30, 300)
       self.fc2 = torch.nn.Linear(300,10)
   def forward(self, x ):
       x = F.relu(self.conv1(x))
       x = F.max_pool2d(x, 2, 2)
       x = F.relu(self.conv2(x))
       x = F.max_pool2d(x, 2, 2)
       x = x.view(-1, 5*5*30)
       x = F.relu(self.fc1(x))
       #x = F.log_softmax(self.fc2(x))
       x = self.fc2(x)
       return x

model = StudentNet().to(device)
# optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                     weight_decay=5e-4)

def train(epoch, model):
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output,target)
        loss.backward()
        optimizer.step()
        if batch_index % args.log_interval == 0:
           print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_index * len(data), len(train_loader.dataset),
                100. * batch_index / len(train_loader), loss.item())) 


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1,keepdim=True)          # get the index of the max-logprobability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\n Test set: Avarage loss: {:.6f} accuracy: {}/{} ({:.0f}%)'.format(
             test_loss, correct, len(test_loader.dataset),
             100. * correct/len(test_loader.dataset))) 



for epoch in range(1, args.epochs + 1):
    train(epoch, model)
    test(model)
if args.save_model:
    torch.save(model.state_dict(),'student_mnist_cnn_linear.pt')
print("----second-----: ",time.time() - start_time)
