#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:59:51 2019

@author: emilyjones
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt 
import torch
from torch import nn


data = np.load('JPmarket_dataset.npz')

train_ratios = data['train_ratios']
test_ratios = data['test_ratios']
#train_vols = data['train_volumes']
#test_vols = data['test_volumes']

def getbd_from_theta(theta): 
  gamma, beta, delta = theta
  L = len(beta)
  b = torch.zeros(L+1)
  for l in range(L): 
    if l == 0 : 
      b[l] = beta[l]
    else: 
      b[l] = beta[l]-beta[l-1]
  d = torch.zeros(L+1)
  for l in range(1,L+1): 
    d[l] = torch.sum(delta[:l])
  return b, d

def crps_loss(theta, z):
  gamma, beta, delta = theta
  L = len(beta)
  b, d = getbd_from_theta(theta)
  
  zeros = torch.zeros(L+1)
  
  lo = 0 
  for l in range(L, -1, -1): 
    val = sqf(theta, d[l])
    if val < z:
      lo = l
      break 
  
  a_tilde = (z-gamma + torch.sum(b[:lo+1]*d[:lo+1]))/torch.sum(b[:lo+1])
  max_ = torch.max(zeros+a_tilde, d)
  bracket = (1/3)*(1-torch.pow(d, 3)) - d - torch.pow(max_,2) + 2*max_*d
  loss = (2*a_tilde - 1)*z + (1-2*a_tilde)*gamma + torch.sum(b*bracket)
  
  return loss

def sqf(theta, quantile): 
  
  gamma, beta, delta = theta
  L = len(beta)
  b,d = getbd_from_theta(theta)
  max_ = torch.max(quantile-d, torch.zeros(L+1))
  qf = gamma + torch.sum(b*max_)
  
  return qf

#covariate function 
#inputs - input_data of shape [number of time series, length of each time series], freq of data i.e. number of bins per day, and position of special bins as a vector
#returns - data and covariate vector of shape [number of series, length of series, number of special bins + 2] where covariate_vectors[:,:,0] is the input data, 
#covariate_vectors[:,:,1] is the scaled time of day and covariate_vectors[:,:,2:] is the one hot vector for the special bins
#therefore covariate_vectors[:,:,0] is the data and covariate_vectors[:,:,1:] is the actual covariate vector 

#for this model we actually need to input the observation at time t with the covariate at the next time step 
#so here we create covariates one step ahead 
def new_create_covariate_data(input_data, freq, pos_of_special_bins): 
  num_series, len_series = input_data.shape
  days = int(len_series/freq)
  num_special_bins = len(pos_of_special_bins)
  covariate_vectors = np.zeros((num_series, len_series, num_special_bins+2))
  
  for n in range(num_series): 
    for d in range(days): 
      for t in range(freq): 
        x = np.zeros(num_special_bins + 1)
        #x[0] is the scaled time of day 
        x[0] = (t+1)/freq
        
        check = (t+1) in pos_of_special_bins
        
        if check == True: 
          index = pos_of_special_bins.index(t+1)
          x[index+1] = 1
          
        covariate_vectors[n, d*freq + t, 0] = input_data[n, d*freq + t]
        covariate_vectors[n, d*freq + t, 1:] = x
        
  return covariate_vectors

class Encoder(nn.Module): 
  
  def __init__(self, input_size, hidden_size, batch_size, output_size, num_layers):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.batch_size = batch_size
    self.output_size = output_size
    self.num_layers = num_layers
    
    self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.num_layers)
    self.linear = nn.Linear(self.hidden_size, self.hidden_size)
    self.dense = nn.Linear(self.hidden_size, 1)
    self.softmax = nn.functional.softmax
    self.softplus = nn.functional.softplus
    self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
    self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
    self.linear3 = nn.Linear(self.hidden_size, self.hidden_size)
    
  def init_hidden(self): 
    return [torch.zeros(self.num_layers, 1, self.hidden_size),
            torch.zeros(self.num_layers, 1, self.hidden_size)]
  
  def forward(self, data, hidden):
    lstm_out, hidden = self.lstm(data.view(1, 1, -1), hidden)
    fc_layer = self.linear(lstm_out.view(-1)) 
    
    lin1 = self.linear1(fc_layer)
    delta = self.softmax(lin1)
    lin2 = self.linear2(fc_layer)
    beta = self.softplus(lin2)
    lin3 = self.linear1(fc_layer)
    gamma = self.dense(lin3)
    
    theta = (gamma, beta, delta)
    
    return theta, hidden

learning_rate = 0.001
#learning_rate_decay = 
num_epochs = 500
T = 2*64
num_paths = 10
pred_length = 64
num_lstm_layers = 5
hidden_units = 40
early_stop_patience = 5
number_series = 50
number_windows = 100
special_bins = [0,31,32,63]
batch_size = 10

#inputs: data of shape (num_series, num_days, num_bins), num_series = number of time series to use, len_series = length of series,  num_windows = number of series to create from each 
#output: data of shape (num_series, num_windows, len_series*bins)

def create_train_ratio_data(data, num_series, len_series, num_windows): 
  total_num_series, num_days, num_bins = data.shape 
  days_series = int(len_series/num_bins)
  
  train_data = np.zeros((num_series, num_windows, len_series))
  
  for i in range(num_series): 
    series_i = data[i]
    for j in range(num_windows):
      start_index = np.random.randint(0,num_days-days_series)
      train_data[i,j] = series_i[start_index:start_index + days_series].flatten()
 
  return train_data

training_ratio = create_train_ratio_data(train_ratios[:,:-int(T/64)], number_series, T, number_windows)
covars_training_data = np.zeros((number_series, number_windows, T, len(special_bins)+2))

for i in range(number_series): 
  covars = new_create_covariate_data(training_ratio[i], 64, special_bins)
  covars_training_data[i] = covars
  
  
covars_train_data = torch.FloatTensor(covars_training_data)

test_ratio = train_ratios[0:number_series].reshape((number_series,-1))
  
new_test_data = test_ratio[:,-T:]

covars_test_data = new_create_covariate_data(new_test_data, 64, [0,31,32,63])

covars_test_data = torch.FloatTensor(covars_test_data)

encoder = Encoder(6, hidden_units, batch_size, 1, num_lstm_layers)

def batch_index_generator(num_series, current_indices):
    index = np.random.randint(0, num_series)
    if index in current_indices: 
        index = batch_index_generator(num_series, current_indices)  
    return index
    

### TRAINING the encoder###

loss_function = crps_loss
encoder_optimiser = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
train_loss = []
test_loss = []
early_stop_count = 0
to_break = False

for i in range(num_epochs): 
  encoder.zero_grad()
  encoder_hidden = encoder.init_hidden()
  
  if to_break: 
      break 
  
  batch_loss = 0
  
  window_index = np.random.randint(0,number_windows)
  window_batch = covars_train_data[:,window_index,:,:]
  
  batch_index = []
  data_batch = torch.zeros((batch_size, T, 6))
  for b in range(batch_size): 
      if b == 0: 
          batch_index.append(np.random.randint(0, number_series))
      else:
          index = batch_index_generator(number_series, batch_index)
          batch_index.append(index)
      current_index = batch_index[b]
      data_batch[b] = window_batch[batch_index[b]]

  
  for j in range(batch_size):
    loss_t = 0
    count = 0 
    #input_data = data_batch[j]
    
    for t in range(T-1): 
      theta, encoder_hidden = encoder(data_batch[j][t:t+1], encoder_hidden)
      loss_t += loss_function(theta, data_batch[j][t+1:t+2,0])
      count += 1
   
    batch_loss += loss_t
    
  encoder_optimiser.zero_grad()
  
  batch_loss.backward()

  encoder_optimiser.step()

  train_loss.append(batch_loss.item())
  
  print("Epoch ", i, "train CRPS ", batch_loss.item())
  
  if i % 10 == 0: 
      with torch.no_grad(): 
        test_data_batch = covars_test_data
        test_batch_loss = 0
        for k in range(batch_size): 
          test_loss_t = 0 
          count = 0 
          #input_test_data = test_data_batch[k]
          for t in range(T-1): 
              theta, encoder_hidden = encoder(test_data_batch[k][t:t+1], encoder_hidden)
              test_loss_t += loss_function(theta, test_data_batch[k][t+1:t+2,0])
              count += 1
              
          test_batch_loss += test_loss_t/count
          
        test_loss.append(test_batch_loss.item())
          
        if i > 100 : 
          if test_loss[-1] > test_loss[-2]: 
              early_stop_count += 1
              if early_stop_count > early_stop_patience: 
                  to_break = True 
          else: 
              early_stop_count = 0
    
      
      print("test CRPS", test_batch_loss.item())
  
  torch.cuda.empty_cache()
  
torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'optimizer_state_dict': encoder_optimiser.state_dict()
            }, 'sqf_ratio_saved.tar')
np.savez('sqf_rnn_ratio_train_loss', train_loss)
np.savez('sqf_rnn_ratio_test_loss', test_loss)

#encoder = Encoder(6, hidden_units, batch_size, 1, num_lstm_layers)
#encoder_optimiser = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
#checkpoint = torch.load('sqf_ratio_saved.tar')
#encoder.load_state_dict(checkpoint['encoder_state_dict'])
#encoder_optimiser.load_state_dict(checkpoint['optimizer_state_dict'])

test_ratio = test_ratios[0:number_series].reshape((number_series,-1))

#windows method 
num_test_windows = 5
  
test_windows = np.zeros((number_series, num_test_windows, T+pred_length))
window_freq = int((test_ratio.shape[1]-(T+pred_length))/num_test_windows)
for j in range(num_test_windows):
    if j == num_test_windows - 1: 
        index = test_ratio.shape[1] - (T+pred_length)
    else: 
        index = j*window_freq
    print(index)
    test_windows[:,j,:] = test_ratio[:,index:index+(T+pred_length)]

covars_test_data = np.zeros((number_series,num_test_windows,(T+pred_length),6))
for i in range(number_series): 
    covars_test_data[i] = new_create_covariate_data(test_windows[i], 64, [0,31,32,63])

input_test_data = torch.FloatTensor(covars_test_data)

## PREDICTION ## 
decoder = encoder
paths = np.zeros((number_series, num_test_windows, num_paths, T + pred_length))
unconstrained_paths = np.zeros((number_series, num_test_windows, num_paths, T + pred_length))
with torch.no_grad(): 
  
  for  j in range(number_series): 
      for k in range(num_test_windows): 
          
        for t in range(T): 
          theta, encoder_hidden = encoder(input_test_data[j][k][t], encoder_hidden)
    
        for n in range(num_paths): 
          decoder_input = input_test_data[j][k][T-1]
          decoder_hidden = encoder_hidden 
          sample_path = torch.zeros(pred_length)
    
          for i in range(pred_length): 
            theta, decoder_hidden = decoder(decoder_input, decoder_hidden)
            alpha = torch.distributions.uniform.Uniform(0,1).rsample()
            z_hat = sqf(theta, alpha)
            sample_path[i] = z_hat
            z_x = torch.zeros(len(special_bins)+2)
            z_x[0] = z_hat
            z_x[1] = (i+1)/64
    
            check = (i+1) in special_bins
            if check == True: 
              index = special_bins.index(i+1)
              z_x[index+2] = 1
    
            decoder_input = z_x
    
          paths[j,k,n,:T] = input_test_data[j][k][:T,0]
          unconstrained_paths[j,k,n,:T] = input_test_data[j][k][:T,0]
          unconstrained_paths[j,k,n, T:] = sample_path.numpy()
          constrained_sample_path = torch.nn.functional.softmax(sample_path)

          paths[j,k,n, T:] = constrained_sample_path.numpy()


#sqf_train_loss = train_loss
#sqf_paths = paths    
np.savez('sqf_ratio_test_data', test_windows)
np.savez('sqf_rnn_test_loss_ratio', test_loss)
np.savez('sqf_rnn_paths_ratio', paths)
np.savez('sqf_rnn_unconstrained_paths_ratio', unconstrained_paths)





