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
        loss = np.linalg.norm(output - data_slice, axis=1).mean()
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
            * np.log(np.linalg.norm(np.divide(out_list[-1] - out_list_0[-1], delta_in)))
        )
    # lamda /= tests
    return np.average(lamda_list)


def config_space(model, noise_factor=10, tests=5, m=20, t=500, convergence=0.01):
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


def rank(model, noise_factor=1, tests=5, m=20, t=500, noise_function=None):
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

def memory(model):
    #MC
    # https://www.cs.bham.ac.uk/~pxt/PAPERS/scr_tnn.pdf section V
    '''
    1.generate input sequence
    2.train model
    3.
    '''
    return None

