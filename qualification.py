import os
import utils
import model
import numpy as np
import torch
import copy

def learning_curve(model_list,test_length,data):
    loss_list = []
    for model in model_list:
        output = model.run_by_self(data)[model.training_length:model.training_length+test_length]
        data_slice = data[model.training_length:model.training_length+test_length]
        loss = np.linalg.norm(output-data_slice,axis=1).mean()
        loss_list.append(loss)
    return loss_list

def lyapunov_estimate(model,initial_in,tests = 100, delta_naught = 0.1, measure_time = 1000):
    lamda = 0
    dimension = model.d_m
    model.training_time = 1
    model.run_time = measure_time
    test_data_0 = [initial_in]
    out_list_0 = model.run_by_self(test_data_0)
    for i in range(tests):
        delta_in = delta_naught*np.random.rand(3)
        test_data = [torch.add(initial_in,delta_in)]
        out_list = model.run_by_self(test_data)
        lamda += 1/measure_time * np.log(np.norm(np.divide(out_list[-1]-out_list_0[-1],delta_in)))
    lamda /= tests
    return lamda

def config_space(model):
    KR,GR,MC=0,0,0

    return KR,GR,MC
        
