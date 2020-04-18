#!/usr/bin/env python3


# refrence: https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter04_DL_computation/4.2_parameters 
# construct a model
import torch
import torch.nn as nn
class Mylistdense(nn.Module):
    def __init__(self):
        super(Mylistdense,self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4,4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4,1)))
    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x

net_1 = Mylistdense()
print(net_1)

class Mydictdense(nn.Module):
    def __init__(self):
        super(Mydictdense, self).__init__()
        self.params = nn.ParameterDict({
            "linear1": nn.Parameter(torch.randn(4, 4)),
            "linear2": nn.Parameter(torch.randn(4, 1)),

        })
        self.params.update({"linear3": nn.Parameter(torch.randn(4,2))})
    def forward(self, x, choice="linear1"):
        return torch.mm(x,self.params[choice])

net_2 = Mydictdense()
print(net_2)


net = nn.Sequential(
    Mylistdense(),
    Mydictdense()
    
)

print("########",net[0])
print(net_2.named_parameters())
for name, params in net_2.named_parameters():
    print("name: ",name, "params:", params)

class Mydense(nn.Module):
    def __init__(self):
        super(Mydense, self).__init__()
        self.linear1 = nn.Linear(4, 4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4, 4)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

    
net_3 = Mydense()
print(net_3)
for name, params in net_3.named_parameters():
    print("name: ",name, "params: ", params)

for p in net_3.parameters():
    print(p.data)

print("####: ",dir(net_3))
print("####:", net_3.state_dict() )