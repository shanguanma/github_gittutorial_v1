#!/usr/bin/env python3
# 2019-12-25 , ma duo
# refrence:https://towardsdatascience.com/the-lstm-reference-card-6163ca98ae87
# This script shows how to use numpy to implement lstm forward propagation

# part1 : using numpy to construct a lstm
import numpy as np
from scipy.special import expit as sigmoid


# The Forget Gate is a way to selectively forget some of what the Cell State (LTM) has in memory. 

# 1. The New Event and the previous period’s Hidden State are summed (element-wise) 
# 2. and then transformed with a sigmoid function. 
# This output is therefore a vector with entries between 0 and 1. 
# 3. When the Previous Cell State is multiplied (element-wise) with this vector, 
# the effect is that some proportion (between 0 and 1) of each value in the 
# Previous Cell State makes it “through the gate” and is retained; the rest is forgotten.

def forget_gate(x, h, Weight_hf, Bias_hf, Weight_xf, Bias_xf, prev_cell_state):
    # x : input of the lstm , size: (x,1)
    # h : previous hidden state of the lstm, size: (h,1)
    # Weight_hf : in forget gate , weight of hidden state of the lstm, size:(h,h)
    # Bias_hf : in forget gate , bias of hidden state of the lstm, size: (h,1)
    # Weight_xf : in forget gate, weight of x of the lstm. size:(h,x)
    # Bias_xf : in forget gate, bias of x of the lstm. size:(h,1)
    # prev_cell_state : previous cell state of the lstm, 
     
    # 1. The New Event and the previous period’s Hidden State are summed (element-wise) 
    # forget_hidden is a vector,size:(h,1)
    # forget_eventx is a vector,size:(h,1)
    # forget_merge is a vector, size:(h,1)
    forget_hidden = np.dot(Weight_hf, h) + Bias_hf
    forget_eventx = np.dot(Weight_xf, x) + Bias_xf
    forget_merge =forget_hidden + forget_eventx   
    # 2. transformed with a sigmoid function.
    # forget_merge_sigmoid is a vector, size:(h,1)
    forget_merge_sigmoid = sigmoid(forget_merge)
    # 3. the Previous Cell State is multiplied (element-wise) with this vector,
    # result is a vector ,size: (h,1)
    return np.multiply(forget_merge_sigmoid, prev_cell_state)
    

# The Input Gate has 2 components: a way to “Ignore” new information, 
# and a way to “Learn” new information. In each case, 
# the New Event and previous period’s Hidden State are summed and transformed. 
# The Ignore component is transformed using logic similar to the Forget Gate:
#  a sigmoid function creates a vector of proportions (values between 0 and 1). 
# The Learn component uses a hyperbolic tangent function, 
# which returns a vector with values between -1 and 1; 
# this helps the model learn both positive and negative relationships in the data. 
# When the Learn component is multiplied (element-wise) by the Ignore component, 
# the effect is that some proportion of each value from the Learn component makes 
# it “through the gate” and is retained; the rest is ignored.
def input_gate( x, h, Weight_hi, Bias_hi, Weight_xi, Bias_xi, 
                      Weight_hl, Bias_hl, Weight_xl, Bias_xl): 
    # x : input of the lstm , size: (x,1)
    # h : previous hidden state of the lstm, size: (h,1)
    # Weight_hi : The ignore component in input gate, weight of hidden state of the lstm, size:(h,h)
    # Bias_hi : The ignore component in input gate , bias of hidden state of the lstm, size: (h,1)
    # Weight_xi : The ignore component in input gate, weight of x of the lstm. size:(h,x)
    # Bias_xi : The ignore component in input gate, bias of x of the lstm, size:(h,1)
    # Weight_hl : The learn  component in input gate, weight of hidden state of the lstm, size:(h,h)
    # Bias_hl : The  learn component in input gate , bias of hidden state of the lstm, size: (h,1)
    # Weight_xl : The learn component in input gate, weight of x of the lstm. size:(h,x)
    # Bias_xl : The learn component in input gate, bias of x of the lstm, size:(h,1)
    
    # 1. The Input Gate has 2 components: a way to “Ignore” new information, 
    # and a way to “Learn” new information. In each case, 
    # the New Event and previous period’s Hidden State are summed and transformed. 

    # ignore_hidden is a vector, size:(h,1)
    # ignore_eventx is a vector, size:(h,1)
    # learn_hidden is a vector, size:(h,1)
    # learn_eventx is a vector, size:(h,1)
    ignore_hidden = np.dot(Weight_hi, h) + Bias_hi
    ignore_eventx = np.dot(Weight_xi, x) + Bias_xi
    learn_hidden = np.dot(Weight_hl, h) + Bias_hl
    learn_eventx = np.dot(Weight_xl, x) + Bias_xl
    # ignore_merge is a vector, size:(h,1)
    # learn_merge is a vector, size:(h,1)
    ignore_merge = ignore_hidden + ignore_eventx 
    learn_merge = learn_hidden + learn_eventx
    # 2. transformed with a sigmoid function, in the ignore component
    #    transformed with a hyperbolic tangent function, in the learn component   
      
    # ignore_merge_sigmoid  is a vector , size:(h,1)
    # learn_merge_tanh is a vector, size:(h,1)
    ignore_merge_sigmoid = sigmoid(ignore_merge)
    learn_merge_tanh = np.tanh(learn_merge)
    # 3. multiply
    return np.multiply(ignore_merge_sigmoid, learn_merge_tanh)

# The Cell State at each time is calculated by adding two things together: 
# the vector from the Forget Gate, and the vector from the Input Gate. 
# The Cell State is used in the Output Gate (below) to determine the model’s current output; 
# it’s also carried forward to be used for the next Event’s forward pass.
def cell_state (forget_gate_output, input_gate_output):
    return forget_gate_output + input_gate_output

# The Output Gate returns a vector that is both the model’s output for that Event, 
# and the new hidden state h (STM), which is carried forward to the next Event’s forward pass. 
# The Cell State, previous Hidden State, and New Event all contribute to this vector: 
# the New Event and previous Hidden State are combined and multiplied (element-wise) by the transformed Cell State.
def output_gate(x, h, Weight_ho, Bias_ho, Weight_xo, Bias_xo, cell_state):
    # x : input of the lstm , size: (x,1)
    # h : previous hidden state of the lstm, size: (h,1)
    # Weight_ho : in output gate, weight of hidden state of the lstm, size:(h,h)
    # Bias_ho : in output gate , bias of hidden state of the lstm, size: (h,1)
    # Weight_xo : in output gate, weight of x of the lstm. size:(h,x)
    # Bias_xo : in output gate, bias of x of the lstm, size:(h,1)
    
    # 1. 
    # out_hidden is a vector, size:(h,1)
    # out_eventx is a vector, size:(h,1)
    # out_merge is a vector, size:(h,1)
    out_hidden = np.dot(Weight_ho, h) + Bias_ho
    out_eventx = np.dot(Weight_xo, x) + Bias_xo
    out_merge = out_hidden  + out_eventx
    # 2. transformed with a sigmoid function
    # out_merge_sigmoid is a vector, size:(h,1)
    out_merge_sigmoid = sigmoid(out_merge)
    # 3. multiply
    # the result is a vector, size:(h,1)
    return np.multiply(out_merge_sigmoid, np.tanh(cell_state))

# Typically, an LSTM feeds a final, fully-connected linear layer. Let’s do that as well:
# set paremeters for a small lstm network
input_size = 2 # size of one "event", or sample, 
hidden_dim = 3 # 3 cells in the lstm layer
output_size =1 # desired model output

def model_output(lstm_output, fc_Weight, fc_Bias):
    # take the lstm output and transformes it to our desired output size using a final, full connected layer
    return np.dot(fc_Weight, lstm_output) + fc_Bias


# part2
# Create a PyTorch LSTM with the same parameters. 
# PyTorch will automatically assign the weights with random 
# values — we’ll extract those and use them to initialize our NumPy network as well.

import torch
import torch.nn as nn

# initialize an Pytorch lstm for comparsion tp our Numpy LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, n_layers=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        # a Final, full connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    def forward(self, x, hidden):
        batch_size = 1
        # get LSTM output
        lstm_output, (h, c) = self.lstm(x, hidden)
        # output shape is to be (batch*seq_lenght, hidden_dim)
        lstm_output = lstm_output.view(-1, self.hidden_dim)
       
        # get final output
        model_output = self.fc(lstm_output)

        return model_output,(h,c)

torch.manual_seed(5)

torch_lstm = LSTM(input_size=input_size,
                  hidden_dim=hidden_dim,
                  output_size=output_size)

state = torch_lstm.state_dict()

# The weights for each gate  are in this order: ignore, forget, learn, output
print(state)
# Given the parameters we chose, we can therefore extract the weights for the NumPy LSTM to use in this way:

# event (x) Weights and Biases for all gates
Weight_xi = state['lstm.weight_ih_l0'][0:3].numpy() # shape (h,x) for ignore component in input gate
Weight_xf = state['lstm.weight_ih_l0'][3:6].numpy() # shape (h,x) for forget gate
Weight_xl = state['lstm.weight_ih_l0'][6:9].numpy() # shape (h,x) for learn component in input gate
Weight_xo = state['lstm.weight_ih_l0'][9:12].numpy() # shape (h,x) for  output gate

Bias_xi = state['lstm.bias_ih_l0'][0:3].numpy()  #shape is [h, 1] for ignore component in input gate
Bias_xf = state['lstm.bias_ih_l0'][3:6].numpy()  #shape is [h, 1] for forget gate
Bias_xl = state['lstm.bias_ih_l0'][6:9].numpy()  #shape is [h, 1] for learn component in input gate
Bias_xo = state['lstm.bias_ih_l0'][9:12].numpy()  #shape is [h, 1] for output gate

# hidden state (h) Weights and Biases for all gates
Weight_hi = state['lstm.weight_hh_l0'][0:3].numpy() # shape (h,x) for ignore component in input gate
Weight_hf = state['lstm.weight_hh_l0'][3:6].numpy() # shape (h,x) for forget gate
Weight_hl = state['lstm.weight_hh_l0'][6:9].numpy() # shape (h,x) for learn component in input gate
Weight_ho = state['lstm.weight_hh_l0'][9:12].numpy() # shape (h,x) for  output gate

Bias_hi = state['lstm.bias_hh_l0'][0:3].numpy()  #shape is [h, 1] for ignore component in input gate
Bias_hf = state['lstm.bias_hh_l0'][3:6].numpy()  #shape is [h, 1] for forget gate
Bias_hl = state['lstm.bias_hh_l0'][6:9].numpy()  #shape is [h, 1] for learn component in input gate
Bias_ho = state['lstm.bias_hh_l0'][9:12].numpy()  #shape is [h, 1] for output gate

# final , full connected layers Weights and Biases
fc_Weight = state['fc.weight'][0].numpy() #shape is [h, output_size]
fc_Bias = state['fc.bias'][0].numpy() #shape is [,output_size]


# Now, we have two networks — one in PyTorch, one in NumPy 
# — with access to the same starting weights. 
# We’ll put some time series data through each to ensure they are identical. 
# To do a forward pass with our network, 
# we’ll pass the data into the LSTM gates in sequence, and print the output after each event:

# Simple time series data
data = np.array([[1,1],
                 [2,2],
                 [3,3]])

# Initialize cell and hidden states with zeros
c = np.zeros(hidden_dim)
h = np.zeros(hidden_dim)

# Loop through data, updating the hidden and cell states after each pass
for eventx in data:
    f = forget_gate(eventx, h, Weight_hf, Bias_hf, Weight_xf, Bias_xf, c)
    i = input_gate(eventx, h, Weight_hi, Bias_hi, Weight_xi, Bias_xi,
                              Weight_hl, Bias_hl, Weight_xl, Bias_xl)
    c = cell_state(f, i)
    h = output_gate(eventx, h, Weight_ho, Bias_ho, Weight_xo, Bias_xo, c)
    print("numpy_lstm_output: {0}".format(model_output(h, fc_Weight, fc_Bias)))

# pytorch expects an extral dimension for batch size
torch_data = torch.Tensor(data).unsqueeze(0)
torch_output, (torch_hidden, torch_cell) = torch_lstm(torch_data, None)
print("torch_lstm_output: \n",torch_output)

# check

print("\n",'-'*40)
print(f'torch hidden state: {torch_hidden}')
print(f'torch cell state: {torch_cell}')
print(f'numpy hidden state: {h}')
print(f'numpy cell state: {c}')
