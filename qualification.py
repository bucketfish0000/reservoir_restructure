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

def lyapunov(model):
    return

def config_space(model):
    KR,GR,MC=0,0,0

    return KR,GR,MC
        
