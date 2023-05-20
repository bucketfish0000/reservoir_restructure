import os
import utils
import model
import numpy as np
import torch
import copy

def learning_curve(model_list,test_length,data):
    loss_list = []
    for model in model_list:
        output,_ = model.run(data,test_length,0)
        #print(output.shape)
        data_slice = data
        loss = np.linalg.norm(output-data_slice,axis=1).mean()
        loss_list.append(loss)
    return loss_list

def lyapunov_estimate(model,initial_in,tests = 100, delta_naught = 0.1, measure_time = 1000):
    lamda = 0
    dimension = model.d_m
    test_data_0 = [initial_in]
    out_list_0,_ = model.run(test_data_0,1,measure_time)
    for i in range(tests):
        delta_in = delta_naught*np.random.rand(4)
        delta_in[3] =0
        test_data = [np.add(initial_in,delta_in)]
        out_list,_ = model.run(test_data,1,measure_time)
        #print(out_list[-1]-out_list_0[-1])
        lamda += 1/measure_time * np.log(np.linalg.norm(np.divide(out_list[-1]-out_list_0[-1],delta_in)))
    lamda /= tests
    return lamda

def config_space(model,noise_factor = 10, tests = 20, matrix_size = 500):
    KR,GR,MC=0,0,0
    #feed and measure random seq of input
    inputs = np.random.rand(tests*matrix_size,model.d_m)
    noisy_inputs = noise_factor*np.random.rand(tests*matrix_size,model.d_m)
    outputs,subsamples = model.run(inputs,tests*matrix_size,0)
    #KR
    for i in np.linspace(0,len(outputs),num=tests):
        if i==len(outputs): break
        kernel = subsamples[round(i):round(i)+matrix_size]
        KR += np.linalg.matrix_rank(kernel)
    KR /=tests
    noisy_outputs, noisy_subsamples = model.run(noisy_inputs,tests*matrix_size,0)
    #GR
    for i in np.linspace(0,len(outputs),num=tests):
        if i==len(outputs): break
        kernel = noisy_subsamples[round(i):round(i)+matrix_size]
        GR += np.linalg.matrix_rank(kernel)
    GR /=tests
    #MC

    return KR,GR,MC

def determination(v1,v2):
    return (np.cov([v1,v2])**2)/((np.var(v1)**2)*(np.var(v2)**2))


