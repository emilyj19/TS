#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:51:07 2019

@author: emilyjones
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:18:53 2019

@author: emilyjones
"""

import numpy as np 
import torch
from torch import nn

data = np.load('JPmarket_dataset.npz')

train_ratios = data['train_ratios']
test_ratios = data['test_ratios']
train_vols = data['train_volumes']
test_vols = data['test_volumes']

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
        x[0] = t/freq
        
        check = t in pos_of_special_bins
        
        if check == True: 
          index = pos_of_special_bins.index(t)
          x[index+1] = 1
          
        covariate_vectors[n, d*freq + t, 0] = input_data[n, d*freq + t]
        covariate_vectors[n, d*freq + t, 1:] = x
        
  return covariate_vectors

number_series = 50
number_windows = 200
T = 20*64
special_bins = [0,31,32,63]

training_ratio = create_train_ratio_data(train_ratios[:,:-int(T/64)], number_series, T, number_windows)
covars_training_data = np.zeros((number_series, number_windows, T, len(special_bins)+2))

for i in range(number_series): 
  covars = new_create_covariate_data(training_ratio[i], 64, special_bins)
  covars_training_data[i] = covars
  
  
covars_train_data = torch.FloatTensor(covars_training_data)

#Test data
test_ratio = train_ratios[0:number_series].reshape((number_series,-1))
  
new_test_data = test_ratio[:,-T:]

covars_test_data = new_create_covariate_data(new_test_data, 64, [0,31,32,63])

covars_test_data = torch.FloatTensor(covars_test_data)

def init_weights(m): 
    if isinstance(m, nn.Linear): 
        nn.init.xavier_uniform_(m.weight.data)
    elif isinstance(m, nn.LSTM) or isinstance(m, nn.RNN):
        nn.init.xavier_uniform_(m.weight_ih_l0.data)
        nn.init.xavier_uniform_(m.weight_hh_l0.data)

class GlobalEffects(nn.Module): 
  def __init__(self, input_size, num_factors, hidden_size, batch_size, output_size = 1, num_layers = 1): 
    super().__init__()
    self.input_size = input_size
    self.num_factors = num_factors
    self.hidden_size = hidden_size
    self.batch_size = batch_size
    self.output_size = output_size
    self.num_layers = num_layers
    
    self.linears = nn.ModuleList([nn.Linear(self.hidden_size, self.output_size, bias = False) for i in range(self.num_factors)])
    self.lstms = nn.ModuleList([nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.num_layers) for i in range(self.num_factors)])
    
    self.linears.apply(init_weights)
    self.lstms.apply(init_weights)
    
    self.w = torch.nn.Parameter(torch.zeros(batch_size, num_factors))
    
  def init_hidden(self): 
    hidden = [torch.zeros(self.num_layers, self.batch_size, self.hidden_size) for i in range(self.num_factors)]
    return hidden
    
  def forward(self, input_data, hidden): 
    x = input_data[:,:,1:]
    for i in range(self.num_factors): 
      lstm_out, hidden[i] = self.lstms[i](x.view(x.shape[1], self.batch_size, -1))
      g_i = self.linears[i](lstm_out).view(1, self.batch_size, -1) #shape of g_i = [1, batch_size, seq_len]
      
      if i == 0: 
        g = g_i
      else:
        g = torch.cat((g,g_i), dim=0)

    fixed_effects = torch.zeros((self.batch_size, g.shape[2]))
    
    for i in range(self.batch_size): 
      for j in range(g.shape[2]): 
        fixed_effects[i,j] = torch.dot(self.w[i], g[:,i,j])
      
    return fixed_effects

class DF_RNN(nn.Module): 
  def __init__(self, input_size, hidden_size, batch_size, num_series, output_size):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.batch_size = batch_size
    self.output_size = output_size
    self.num_series = num_series
    
    self.rnns = nn.ModuleList([nn.RNN(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = 1) for i in range(self.num_series)])
    self.linears = nn.ModuleList([nn.Linear(self.hidden_size, self.output_size) for i in range(self.num_series)])
    self.relu = nn.ReLU()
    
    self.linears.apply(init_weights)
    self.rnns.apply(init_weights)
    
  def init_hidden(self): 
    hidden = [torch.zeros(1, self.batch_size, self.hidden_size) for i in range(self.num_series)]
    return hidden
    
  def forward(self, input_data, hidden, fixed_effects, gaussian_likelihood, prediction): 
    z = input_data[:,:,0]
    x = input_data[:,:,1:]
    
    sigma = torch.zeros((self.num_series, x.shape[1]))
    r = torch.zeros(sigma.shape)
    
    for i in range(self.num_series): 
      data = x[i]
      rnn_out, hidden[i] = self.rnns[i](data.view(data.shape[0], self.batch_size, -1))
      sig = self.linears[i](rnn_out).view(-1)
      
      sigma[i] = torch.abs(sig)
      
      for j in range(sigma.shape[1]):
        r[i,j] = torch.distributions.normal.Normal(0, sigma[i,j]).sample()
        
    u = fixed_effects + r
    u_out= torch.nn.functional.softmax(u)
        
    if prediction == False: 
    
      if gaussian_likelihood == True: 
        log_lik = self.log_likelihood_Gaussian(z, fixed_effects, sigma)

      else: 
        pass
      
      return log_lik, sigma
    
    else: 
      return u_out #sigma
  
  def log_likelihood_Gaussian(self, z, f, sigma):
    log_p = torch.zeros(sigma.shape)
    
    for i in range(sigma.shape[0]):
      for j in range(sigma.shape[1]): 
        log_pdf = torch.distributions.normal.Normal(0, sigma[i,j]).log_prob(z[i,j] - f[i,j])   
        #scale the likelihood to 0-1
        log_norm_constant = torch.distributions.normal.Normal(0, sigma[i,j]).log_prob(0)
        log_p[i,j] = log_pdf - log_norm_constant
    
    log_lik = torch.sum(log_p)
#    log_lik = torch.sum(log_p)

    return log_lik

learning_rate = 0.0001
batch_size = number_series
num_epochs = 500
hidden_units_global = 60
hidden_units_local = 5
n_factors = 10 
early_stop_patience = 5

global_model = GlobalEffects(input_size = 5, num_factors = n_factors , hidden_size = hidden_units_global, batch_size= number_series)  
local_model = DF_RNN(5, hidden_size = hidden_units_local, batch_size = 1, num_series = number_series, output_size = 1)


optimiser = torch.optim.SGD(list(global_model.parameters()) + list(local_model.parameters()), lr = learning_rate)

train_loss = []
test_loss = []

to_break= False
early_stop_count = 0
 
for t in range(num_epochs): 
    global_model.zero_grad()
    global_hidden = global_model.init_hidden()
  
    local_model.zero_grad()
    local_hidden = local_model.init_hidden()
    
    if to_break: 
      break 
  
    batch_index = np.random.randint(0,number_windows)
    data_batch = covars_train_data[:,batch_index,:,:].contiguous()
  
    fixed_effects = global_model(data_batch, global_hidden)

    log_lik, sigma = local_model(data_batch, local_hidden, fixed_effects, gaussian_likelihood = True, prediction = False)

    batch_loss = -1*log_lik
    
    optimiser.zero_grad()
  
    batch_loss.backward()

    optimiser.step()
  
    train_loss.append(batch_loss.item())
    
    #print("Epoch: ", t, "loss: ", batch_loss.item())
    
#    if np.isnan(train_loss[t]): 
#        np.savez('DF_train_loss', train_loss)
#        np.savez('DF_test_loss', test_loss)
#        exit()
  
    if t % 10 == 0: 
        with torch.no_grad():
            test_data_batch = covars_test_data
            fixed_effects = global_model(test_data_batch, global_hidden)
            test_input_data = test_data_batch
            log_lik, sigma = local_model(test_input_data, local_hidden, fixed_effects, gaussian_likelihood = True, prediction = False)
            test_batch_loss = -1*log_lik
            test_loss.append(test_batch_loss.item())
            
            if t > 0 : 
              if test_loss[-1] > test_loss[-2]: 
                  early_stop_count += 1
                  if early_stop_count > early_stop_patience: 
                      to_break = True 
              else: 
                  early_stop_count = 0
            
        #print('testloss', test_batch_loss.item())
        
    torch.cuda.empty_cache()

        
 #   print("Epoch: ", t, "loss: ", batch_loss.item())#, "test loss: ", test_batch_loss.item())


num_paths = 500
pred_length = 64

#test data
test_ratio = test_ratios[0:number_series].reshape((number_series,-1))
num_test_windows = 5
  
test_windows = np.zeros((number_series, num_test_windows, T+pred_length))
window_freq = int((test_ratio.shape[1]-(T+pred_length))/num_test_windows)
for j in range(num_test_windows):
    if j == num_test_windows - 1: 
        index = test_ratio.shape[1] - (T+pred_length)
    else: 
        index = j*window_freq
    test_windows[:,j,:] = test_ratio[:,index:index+(T+pred_length)]

covars_test_data = np.zeros((number_series,num_test_windows,(T+pred_length),6))
for i in range(number_series): 
    covars_test_data[i] = new_create_covariate_data(test_windows[i], 64, [0,31,32,63])

covars_test_data = torch.FloatTensor(covars_test_data)

## PREDICTION ##
paths = np.zeros((number_series, num_test_windows, num_paths, T+ pred_length))

with torch.no_grad(): 
  for i in range(num_test_windows): 
      test_data = covars_test_data[:,i,-pred_length:,:].contiguous()
      
      for n in range(num_paths): 
        
        fixed_effects = global_model(test_data, global_hidden)
       
        u = local_model(test_data, local_hidden, fixed_effects, gaussian_likelihood = True, prediction = True) 
        
        paths[:,i,n,:T] = covars_test_data[:,i,:T,0]
        paths[:,i,n, -pred_length:] = u

#results_filename = 'DF_vol_paths'
np.savez('DF_test_data', test_windows)
np.savez('DF_train_loss', train_loss)
np.savez('DF_test_loss', test_loss)
np.savez('DF_vol_paths', paths)
            
            


