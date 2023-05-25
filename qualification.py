import os
import utils
import model
import numpy as np
import torch
import copy


def learning_curve(model_list, test_length, data):
    loss_list = []
    for model in model_list:
        output, _ = model.run(data, test_length, 0)
        # print(output.shape)
        data_slice = data
        loss = utils.MSE(output,data_slice)
        #print(loss)
        loss_list.append(loss)
    return loss_list


def lyapunov_estimate(
    model, initial_in, tests=100, delta_naught=1e-5, measure_time=1000, dimension=4
):
    # lamda = 0
    dimension = model.d_m
    test_data_0 = [initial_in]
    out_list_0, _ = model.run(test_data_0, 1, measure_time)
    lamda_list = []
    for i in range(tests):
        delta_in = delta_naught * np.random.rand(dimension)
        test_data = [np.add(initial_in, delta_in)]
        out_list, _ = model.run(test_data, 1, measure_time)
        # print(out_list[-1]-out_list_0[-1])
        lamda_list.append(
            1
            / measure_time
            * np.log(np.linalg.norm(np.divide(out_list[-1][:dimension-1] - out_list_0[-1][:dimension-1], delta_in[:dimension-1])))
        )
    # lamda /= tests
    return np.average(lamda_list)


def config_space(model, noise_factor=10, tests=5, m=200, t=50, convergence=0.01):
    KR, GR, MC = 0, 0, 0
    # feed and measure random seq of input
    KR = rank(model, noise_factor=1, tests=tests, m=m, t=t, noise_function=None)
    print("---")
    GR = rank(model, noise_factor=5, tests=tests, m=m, t=t, noise_function=np.square)
    return KR, GR, MC


def random_feed(model, length, scale, noise_function=None):
    # feeds one random input of length <length> through model <model>
    inputs = scale * np.random.rand(length, model.d_m)
    if noise_function != None:
        inputs = noise_function(inputs)
    outputs, subsamples = model.run(inputs, length, 0)
    # return output sequence and subsample sequence of model reservoir states
    return outputs, subsamples


def rank(model, noise_factor=1, tests=5, m=200, t=50, noise_function=None):
    rank_list = []
    for i in range(tests):
        kernel = []
        for j in range(m):
            out, sub = random_feed(
                model, length=t, scale=noise_factor, noise_function=noise_function
            )
            kernel.append(np.array(sub[-1]))
        #print(kernel)
        rank = np.linalg.matrix_rank(kernel)
        print(rank)
        rank_list.append(rank)
    return np.average(rank_list)

def memory(model,data,max_k=100):
    # TODO
    #MC
    # https://www.cs.bham.ac.uk/~pxt/PAPERS/scr_tnn.pdf section V
    '''
    1.generate input sequence
    2.train model
    3.
    '''
    MC=0
    for k in range(1,max_k):
        model_copy = copy.deepcopy(model)
        input = data[k:]
        reference = data[:len(data)-k]
        MC_k = memory_k(model_copy,input,reference,k)
        MC+=MC_k
    return MC

def memory_k(model,training_data,expected_data,k):
    #TODO
    '''
    training input: input[t0,t']
    training reference: input[t0-k,t'-k] (???) 
    '''
    #do training
    _,_,_= model.training(training_data,expected_data,qualification=0)
    #calculate correlation coefficient
    output,_= model.run(training_data,len(training_data),0)
    correlation_list = []
    for t in range(k,len(training_data)):
        corr = correlation(expected_data[t],output[t],training_data[t])
        correlation_list.append(corr)
    return np.average(correlation_list)

def correlation(in_t_minus_k,out_t,in_t):
    cov_1 = np.cov(in_t_minus_k,out_t)
    cov_2 = np.cov(in_t,out_t)
    return (cov_1[0,1])**2/(cov_2[0,0]*cov_2[1,1])


