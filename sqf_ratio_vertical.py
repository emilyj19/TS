#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:45:49 2019

@author: emilyjones
"""

import numpy as np 
import matplotlib.pyplot as plt 
import torch
from torch import nn


data = np.load('JPmarket_dataset.npz')

train_ratios = data['train_ratios']
test_ratios = data['test_ratios']
#train_vols = data['train_volumes']
#test_vols = data['test_volumes']

#need to change this to create windows for all stocks 
def split_data(n_days, data): #data for one stock in the form [days, bins]
  no_ts = data.shape[0]-n_days
  length_ts = n_days*data.shape[1]
  new_data = np.zeros((no_ts,length_ts))
  for j in range(no_ts):
    for i in range(n_days): 
      new_data[j,64*i:64*i + 64] = data[j+i,:]
  
  return new_data

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
  bins,seq_len, L = beta.shape
  loss_av = 0
  for t in range(bins):
      for j in range(seq_len):
          zeros = torch.zeros(L+1)
          theta_new = (gamma[t,j],beta[t,j],delta[t,j])
          b, d = getbd_from_theta(theta_new)
          lo = 0 
          for l in range(L, -1, -1): 
            val = sqf(theta_new, d[l])
            if val < z[0,t]:
              lo = l
              break 
          
          a_tilde = (z[j,t]-gamma[t,j] + torch.sum(b[:lo+1]*d[:lo+1]))/torch.sum(b[:lo+1])
          max_ = torch.max(zeros+a_tilde, d)
          bracket = (1/3)*(1-torch.pow(d, 3)) - d - torch.pow(max_,2) + 2*max_*d
          loss = (2*a_tilde - 1)*z[j,t] + (1-2*a_tilde)*gamma[t,j] + torch.sum(b*bracket)
          loss_av += loss
      
  return loss_av/(bins*seq_len)


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
  num_windows, num_days, bins = input_data.shape
  num_special_bins = len(pos_of_special_bins)
  covariate_vectors = np.zeros((num_windows, num_days, bins, num_special_bins+2))
  
  for t in range(bins):
      x = np.zeros((num_windows, num_days, num_special_bins + 1))
      #x[0] is the scaled time of day 
      x[:,:,0] = (t+1)/bins
      check = (t+1) in pos_of_special_bins
      if check == True: 
         index = pos_of_special_bins.index(t+1)
         x[:,:,index+1] = 1
      covariate_vectors[:,:,t,0] = input_data[:,:,t]
      covariate_vectors[:,:,t,1:] = x
      
  return covariate_vectors

class Encoder(nn.Module): 
  
  def __init__(self, input_size, hidden_size, batch_size, output_size, num_layers):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.batch_size = batch_size
    self.output_size = output_size
    self.num_layers = num_layers
    
    self.lstms = nn.ModuleList([nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.num_layers) for i in range(64)])
    self.linears = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for i in range(64)])

    
    self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.num_layers)
    self.linear = nn.Linear(self.hidden_size, self.hidden_size)
    self.dense = nn.Linear(self.hidden_size, 1)
    self.softmax = nn.functional.softmax
    self.softplus = nn.functional.softplus
    self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
    self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
    self.linear3 = nn.Linear(self.hidden_size, self.hidden_size)
    
  def init_hidden(self): 
    hidden = [[torch.zeros(self.num_layers, 1, self.hidden_size),torch.zeros(self.num_layers, 1, self.hidden_size)] for i in range(64)]
    return hidden
  
  def forward(self, data, hidden):
    fc_layer_seq = []
    for i in range(64): 
        lstm_out, hidden[i] = self.lstms[i](data[:,i].view(10,1,-1))
        fc_layer_i = self.linears[i](lstm_out.view(1,10,-1))
        fc_layer_seq.append(fc_layer_i)
    
    fc_layer = torch.cat(fc_layer_seq, dim = 0)
    
    #lstm_out, hidden = self.lstm(data.view(9, 64, -1), hidden)
    #fc_layer = self.linear(lstm_out.view(64,9,-1)) 
    
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
T = 11*64
num_paths = 10
pred_length = 64
num_lstm_layers = 4
hidden_units = 60
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
  
  train_data = np.zeros((num_series, num_windows, days_series, 64))
  
  for i in range(num_series): 
    series_i = data[i]
    for j in range(num_windows):
      start_index = np.random.randint(0,num_days-days_series)
      train_data[i,j] = series_i[start_index:start_index + days_series]
 
  return train_data


training_ratio = create_train_ratio_data(train_ratios[:,:-int(T/64)], number_series, T, number_windows)
covars_training_data = np.zeros((number_series, number_windows, int(T/64),64, len(special_bins)+2))

for i in range(number_series): 
  covars = new_create_covariate_data(training_ratio[i], 64, special_bins)
  covars_training_data[i] = covars
  
  
covars_train_data = torch.FloatTensor(covars_training_data)

test_ratio = train_ratios[0:number_series]
  
new_test_data = test_ratio[:,-int(T/64):]

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
      torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'optimizer_state_dict': encoder_optimiser.state_dict(), 
        'epoch': i,
        'train_loss': train_loss, 
        'test_loss': test_loss
        }, 'sqf_ratio_vert_saved.tar')
      break 
  
  batch_loss = 0
  
  window_index = np.random.randint(0,number_windows)
  window_batch = covars_train_data[:,window_index,:,:]
  
  batch_index = []
  data_batch = torch.zeros((batch_size, int(T/64),64, 6))
  for b in range(batch_size): 
      if b == 0: 
          batch_index.append(np.random.randint(0, number_series))
      else:
          index = batch_index_generator(number_series, batch_index)
          batch_index.append(index)
      current_index = batch_index[b]
      data_batch[b] = window_batch[batch_index[b]]

  
  for j in range(batch_size):
#    print('batch', j)
    loss_t = 0
    count = 0 
    #input_data = data_batch[j]
    t = int(T/64)-1
    theta, encoder_hidden = encoder(data_batch[j][:t], encoder_hidden)
    loss_t += loss_function(theta, data_batch[j][1:t+1,:,0])
    count += 1
       
    batch_loss += loss_t/count
    
  encoder_optimiser.zero_grad()
  
  batch_loss.backward()

  encoder_optimiser.step()

  train_loss.append(batch_loss.item())
  
  print("Epoch ", i, "train CRPS ", batch_loss.item())
  if i % 90 == 0 and i > 0: 
      torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'optimizer_state_dict': encoder_optimiser.state_dict(), 
        'epoch': i,
        'train_loss': train_loss, 
        'test_loss': test_loss
        }, 'sqf_ratio_vert_saved.tar')
  
  if i % 10 == 0: 
      with torch.no_grad(): 
        test_data_batch = covars_test_data
        test_batch_loss = 0
        for k in range(batch_size): 
          test_loss_t = 0 
          count = 0 
          #input_test_data = test_data_batch[k]
          t = int(T/64)-1
          theta, encoder_hidden = encoder(test_data_batch[k][:t], encoder_hidden)
          test_loss_t += loss_function(theta, test_data_batch[k][1:t+1,:,0])
          count += 1
              
          test_batch_loss += test_loss_t/count
          
        test_loss.append(test_batch_loss.item())
          
        if i > 0 : 
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
            }, 'sqf_ratio_vert_saved.tar')
np.savez('sqf_rnn_ratio_vert_train_loss', train_loss)
np.savez('sqf_rnn_ratio_vert_test_loss', test_loss)


encoder = Encoder(6, hidden_units, batch_size, 1, num_lstm_layers)
encoder_optimiser = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

checkpoint = torch.load('sqf_ratio_vert_short_saved.tar')
encoder.load_state_dict(checkpoint['encoder_state_dict'])
encoder_optimiser.load_state_dict(checkpoint['optimizer_state_dict'])

test_ratio = test_ratios[0:number_series]

#windows method 
num_test_windows = 5
  
test_windows = np.zeros((number_series, num_test_windows, int(T/64),64))
window_freq = int(((test_ratio.shape[1])-int(T/64))/num_test_windows)
for j in range(num_test_windows):
    if j == num_test_windows - 1: 
        index = test_ratio.shape[1] - int(T/64)
    else: 
        index = j*window_freq
    print(index)
    test_windows[:,j,:] = test_ratio[:,index:index+int(T/64)]

covars_test_data = np.zeros((number_series,num_test_windows,int(T/64),64,6))
for i in range(number_series): 
    covars_test_data[i] = new_create_covariate_data(test_windows[i], 64, [0,31,32,63])

input_test_data = torch.FloatTensor(covars_test_data)

## PREDICTION ## 
paths = np.zeros((number_series, num_test_windows, num_paths, T))
unconstrained_paths = np.zeros((number_series, num_test_windows, num_paths, T))
with torch.no_grad(): 
  
  for  j in range(number_series): 
      for k in range(num_test_windows): 
        t = int(T/64)-1
        theta, encoder_hidden = encoder(input_test_data[j][k][:t], encoder_hidden) 
        gamma, beta, delta = theta
        for n in range(num_paths): 
          sample_path = torch.zeros(pred_length)
          for i in range(pred_length):
              theta_i = (gamma[i,-1], beta[i,-1], delta[i,-1])
              alpha = torch.distributions.uniform.Uniform(0,1).rsample()
              z_hat = sqf(theta_i, alpha)
              sample_path[i] = z_hat
              
          paths[j,k,n,:T-pred_length] = input_test_data[j][k][:t,:,0].flatten()
          unconstrained_paths[j,k,n,:T-pred_length] = input_test_data[j][k][:t,:,0].flatten()
          unconstrained_paths[j,k,n, T-pred_length:] = sample_path.numpy()
          constrained_sample_path = torch.nn.functional.softmax(sample_path)
          paths[j,k,n, T-pred_length:] = constrained_sample_path.numpy()



#sqf_train_loss = train_loss
#sqf_paths = paths    
np.savez('sqf_ratio_vert_test_data', test_windows)
np.savez('sqf_vert_paths_ratio', paths)
np.savez('sqf_vert_unconstrained_paths_ratio', unconstrained_paths)


unc_paths = np.load('sqf_vert_unconstrained_paths_ratio.npz')
unc_paths = unc_paths['arr_0']
paths.shape

new_paths = np.zeros(unc_paths.shape)
for i in range(unc_paths.shape[0]): 
    for j in range(unc_paths.shape[1]): 
        for k in range(unc_paths.shape[2]):
            new_paths[i,j,k,:-64] = unc_paths[i,j,k,:-64]
            norm_path = np.clip(unc_paths[i,j,k,-64:],0,None)
            norm_sum = np.sum(norm_path)
            new_paths[i,j,k,-64:] = norm_path/norm_sum





