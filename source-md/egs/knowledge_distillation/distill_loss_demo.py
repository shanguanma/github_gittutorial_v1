#!/usr/bin/env python3

import torch
import torch.nn as nn

import torch.nn.functional as F
# smaple number
N = 10
# class number
C = 5

# the softmax output of teacher model
p = torch.softmax(torch.randn(N, C), dim=1)

# the logit output of student model 
s = torch.randn(N, C ,requires_grad = True)

# softmax output of student model, T = 1
q = torch.softmax(s, dim = 1)

# kl loss 
kl_loss = (nn.KLDivLoss()(torch.log(q),p)).sum(dim=0).mean()

# backward
kl_loss.backward(retain_graph=True)
print(' grad using KL DivLoss')
print(s.grad)
print(dir(s))
#print(s)
#print(s.shape)
#print(q)
#print(q.shape)
#print(torch.log(q))
