#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 08:40:00 2019

@author: emilyjones
"""

import numpy as np 
import matplotlib.pyplot as plt 
import torch
from torch import nn


data = np.load('JPmarket_dataset.npz')

#train_ratios = data['train_ratios']
#test_ratios = data['test_ratios']
train_vols = data['train_volumes']
test_vols = data['test_volumes']

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

#inputs: data of shape (num_series, num_days, num_bins), num_series = number of time series to use, len_series = length of series,  num_windows = number of series to create from each 
#output: data of shape (num_series, num_windows, len_series*bins)

def create_train_vols_data(data, num_series, len_series, num_windows): 
  total_num_series, num_days, num_bins = data.shape 
  days_series = int(len_series/num_bins)
  
  train_data = np.zeros((num_series, num_windows, len_series))
  
  for i in range(num_series): 
    series_i = data[i]
    norm_series_i = series_i/np.amax(series_i) #normalise series
    for j in range(num_windows):
      start_index = np.random.randint(0,num_days-days_series)
      train_data[i,j] = norm_series_i[start_index:start_index + days_series].flatten()
 
  return train_data

def init_weights(m): 
    if isinstance(m, nn.Linear): 
        nn.init.xavier_uniform_(m.weight.data)
    elif isinstance(m, nn.LSTM):
        nn.init.xavier_uniform_(m.weight_ih_l0.data)
        nn.init.xavier_uniform_(m.weight_hh_l0.data)
        
def binary_vector(output_pos,T): 
    out = torch.zeros((T,T))
    for j in range(1,T+1): 
        if output_pos+T-j <= T:
            out[j-1,output_pos+T-j-1] = 1
    return out 

class Encoder(nn.Module):
    def __init__(self, input_size, enc_hidden_size, batch_size, num_layers, encoder_len):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = enc_hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.enc_len = encoder_len
    
        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.num_layers)
    
        self.lstm.apply(init_weights)
        
    def init_hidden(self): 
        return [torch.zeros(self.num_layers, 1, self.hidden_size),
                torch.zeros(self.num_layers, 1, self.hidden_size)]
  
    def forward(self, data, hidden):
        encoder_out, encoder_hidden = self.lstm(data.view(self.enc_len, 1, -1), hidden)
        return encoder_out, encoder_hidden

class Attn_Decoder(nn.Module): 
    def __init__(self, input_size, dec_hidden_size, enc_hidden_size, batch_size, num_layers, T_enc):
        super().__init__()
        self.input_size = input_size
        self.enc_size = enc_hidden_size
        self.dec_size = dec_hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.T_enc = T_enc
        
        self.lstm = nn.LSTM(input_size = self.input_size + self.dec_size, hidden_size = self.dec_size, num_layers = self.num_layers)
        self.linear = nn.Linear(self.dec_size, self.dec_size)
        self.softmax = nn.functional.softmax
        self.softplus = nn.functional.softplus
        self.tanh = torch.tanh
        
        self.linear1 = nn.Linear(self.dec_size, self.dec_size)
        self.linear2 = nn.Linear(self.dec_size, self.dec_size)
        self.linear3 = nn.Linear(self.dec_size, self.dec_size)
        self.dense = nn.Linear(self.dec_size, 1)
        
        #for the attention mechanism 
        self.W = nn.Linear(self.enc_size, self.dec_size, bias = False)
        self.U = nn.Linear(self.enc_size, self.dec_size, bias = False)
        self.V = nn.Linear(self.enc_size, 1, bias = False)
        
        #new attention weight
        self.pi = nn.Parameter(torch.FloatTensor(self.T_enc))
        nn.init.normal_(self.pi)
        
        #initializaton
        self.lstm.apply(init_weights)
        self.linear.apply(init_weights)
        self.linear1.apply(init_weights)
        self.linear2.apply(init_weights)
        self.linear3.apply(init_weights)
        self.dense.apply(init_weights)
        self.W.apply(init_weights)
        self.U.apply(init_weights)
        self.V.apply(init_weights)
        
    def forward(self, data, hidden, enc_output, i, return_attn = False): 
        enc_output = enc_output.permute(1,0,2)
        output_pos = i+1
        j_null = output_pos
        delta = binary_vector(output_pos,self.T_enc)
            
        prod = torch.mv(delta, self.pi)
        
        u_h = self.U(enc_output)
        
        for n in range(u_h.shape[2]): 
            u_h[0,:,n] *= prod
        
        attn_score = self.V(self.tanh(self.W(hidden[0][0].view(1,1,-1)) + u_h))
        
        attn_score[0,:j_null,0] = 0 
        
        attn_weights = self.softmax(attn_score, dim=1)
        
        context_vector = torch.sum(attn_weights * enc_output, dim=1)
        
        concat_input = torch.cat((context_vector, data),-1)
        
        dec_out, dec_hidden = self.lstm(concat_input.view(1,1,-1), hidden)
        
        fc_layer = self.linear(dec_out.view(-1)) 
    
        lin1 = self.linear1(fc_layer)
        delta = self.softmax(lin1)
        lin2 = self.linear2(fc_layer)
        beta = self.softplus(lin2)
        lin3 = self.linear1(fc_layer)
        gamma = self.dense(lin3)
    
        theta = (gamma, beta, delta)
        
        if return_attn: 
            return theta, dec_hidden, attn_weights
    
        return theta, dec_hidden
        
        
learning_rate = 0.001
#learning_rate_decay = 
num_epochs = 500
T_encoder = 2*64
T_decoder = 64
T = T_encoder + T_decoder
num_paths = 200
pred_length = T_decoder
num_lstm_layers = 5
enc_hidden_units = 60
dec_hidden_units = 60
early_stop_patience = 5
number_series = 50
number_windows = 100
special_bins = [0,31,32,63]
batch_size = 10

training_vol = create_train_vols_data(train_vols[:,:-int(T/64)], number_series, T, number_windows)
covars_training_data = np.zeros((number_series, number_windows, T, len(special_bins)+2))
for i in range(number_series): 
  covars = new_create_covariate_data(training_vol[i], 64, special_bins)
  covars_training_data[i] = covars
  
covars_train_data = torch.FloatTensor(covars_training_data)

test_volume = train_vols[0:number_series].reshape((number_series,-1))

norm_test_vols = np.zeros(test_volume.shape)
for i in range(number_series): 
  norm_test_vols[i] = test_volume[i]/np.amax(train_vols[i])
  
new_test_data = norm_test_vols[:,-T:]

covars_test_data = new_create_covariate_data(new_test_data, 64, [0,31,32,63])

covars_test_data = torch.FloatTensor(covars_test_data)

def batch_index_generator(num_series, current_indices):
    index = np.random.randint(0, num_series)
    if index in current_indices: 
        index = batch_index_generator(num_series, current_indices)
        
    return index

encoder = Encoder(6, enc_hidden_units, batch_size, num_lstm_layers, T_encoder-1)
decoder = Attn_Decoder(6, dec_hidden_units, enc_hidden_units, batch_size, num_lstm_layers, T_encoder-1)

loss_function = crps_loss
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr= learning_rate)
train_loss = []
test_loss = []
early_stop_count = 0
to_break = False

for i in range(num_epochs):
  print('epoch', i)
  encoder.zero_grad()
  decoder.zero_grad()
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
    #input_data = data_batch[j]
    
    encoder_output, encoder_hidden = encoder(data_batch[j, :T_encoder-1], encoder_hidden)
    decoder_hidden = encoder_hidden
    
    dec_data = data_batch[j, -(T_decoder+1):]
    
    for t in range(T_decoder):             
      theta, decoder_hidden = decoder(dec_data[t:t+1], decoder_hidden, encoder_output, t)
      loss_t += loss_function(theta, dec_data[t+1:t+2,0])
   
    batch_loss += loss_t
    
  optimizer.zero_grad()
  
  batch_loss.backward()

  optimizer.step()

  train_loss.append(batch_loss.item())
  
  print('Epoch: ', i, 'train loss', batch_loss.item())
  
  if i % 10 == 0: 
      with torch.no_grad(): 
          test_data_batch = covars_test_data
          test_batch_loss = 0
          for k in range(batch_size): 
              test_loss_t = 0 
              encoder_output, encoder_hidden = encoder(test_data_batch[k, :T_encoder-1], encoder_hidden)
              decoder_hidden = encoder_hidden
              dec_test_data = test_data_batch[k, -(T_decoder+1):]
              
              for t in range(T_decoder): 
                  theta, decoder_hidden = decoder(dec_test_data[t:t+1], decoder_hidden, encoder_output, t)
                  test_loss_t += loss_function(theta, dec_test_data[t+1:t+2,0])
              
              test_batch_loss += test_loss_t
    
          test_loss.append(test_batch_loss.item())
          
          if i > 0 : 
              if test_loss[-1] > test_loss[-2]: 
                  early_stop_count += 1
                  if early_stop_count > early_stop_patience: 
                      to_break = True 
              else: 
                  early_stop_count = 0
   
      print("test loss", test_batch_loss.item())
      
  torch.cuda.empty_cache()  
  
torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, 'new_model_saved.tar')
  
      
np.savez('new_model_batch_train_loss', train_loss)
np.savez('new_model_batch_test_loss', test_loss)


test_vol = test_vols[0:number_series].reshape((number_series,-1))

num_test_windows = 1
  
new_norm_test_vols = np.zeros(test_vol.shape)
for i in range(number_series): 
  new_norm_test_vols[i] = test_vol[i]/np.amax(train_vols[i])
  
norm_test_windows = np.zeros((number_series, num_test_windows, T))
window_freq = int((test_vol.shape[1]-T)/num_test_windows)
for j in range(num_test_windows):
    if j == num_test_windows - 1: 
        index = test_vol.shape[1] - T
    else: 
        index = j*window_freq
    norm_test_windows[:,j,:] = new_norm_test_vols[:,index:index+T]

covars_test_data = np.zeros((number_series,num_test_windows,T,6))
for i in range(number_series): 
    covars_test_data[i] = new_create_covariate_data(norm_test_windows[i], 64, [0,31,32,63])

input_test_data = torch.FloatTensor(covars_test_data)

## PREDICTION ## 
paths = np.zeros((number_series, num_test_windows, num_paths, T))
saved_attn_weights = torch.zeros((pred_length, T_encoder-1))
count = 0 
with torch.no_grad(): 
    for  j in range(number_series): 
        
        for k in range(num_test_windows): 
            encoder_hidden = encoder.init_hidden()
            encoder_output, encoder_hidden = encoder(input_test_data[j][k][:T_encoder-1], encoder_hidden)

            for n in range(num_paths): 
                decoder_input = input_test_data[j][k][T_encoder-1]
                decoder_hidden = encoder_hidden 
                sample_path = np.zeros(pred_length)
    
                for i in range(pred_length): 
                    theta, decoder_hidden, attn_weights = decoder(decoder_input.view(1,-1), decoder_hidden, encoder_output, i, return_attn = True)
                    saved_attn_weights[i] += attn_weights.view(-1)
                    count += 1
                    alpha = torch.distributions.uniform.Uniform(0,1).rsample()
                    z_hat = sqf(theta, alpha)
                    if z_hat < 0: 
                        z_hat = 0
                        sample_path[i] = z_hat
                    else: 
                        sample_path[i] = z_hat.item()
                    z_x = torch.zeros(len(special_bins)+2)
                    z_x[0] = z_hat
                    z_x[1] = (i+1)/64
    
                    check = (i+1) in special_bins
                    if check == True: 
                        index = special_bins.index(i+1)
                        z_x[index+2] = 1
    
                    decoder_input = z_x
    
                paths[j,k,n,:T_encoder] = input_test_data[j][k][:T_encoder,0]
                paths[j,k,n, T_encoder:] = sample_path

av_attn_weights = saved_attn_weights/count
attn_weights = av_attn_weights.numpy()

np.savez('new_model_single_test_series', norm_test_windows)
np.savez('batch_attn_single_weights', attn_weights)
np.savez('new_model_single_paths', paths)

