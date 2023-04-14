import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math
import torch

def integration_lorenz(params=(10,28,8/3),init=(25,25,25), epoch=300,delta_t=0.1,dimension=3):
    lorenz=solve_ivp(dynamic_lorenz,(0,epoch),(init[0],init[1],init[2]),args=(params[0],params[1],params[2]),dense_output=True)
    time=np.linspace(0,epoch*delta_t,epoch)
    #x,y,z=lorenz.sol(time)
    return lorenz.sol(time),time

def dynamic_lorenz(t,init,a,b,c):
    x,y,z=init
    dx=a*(y-x)
    dy=(b-z)*x-y
    dz=-c*z+x*y
    return dx,dy,dz

def time_dynamic_system(params=(10,28,8/3),init=(25,25,25), epoch=300,delta_t=0.1,dimension=3,system=dynamic_lorenz):
    sequence=solve_ivp(system,(0,epoch),(init[0],init[1],init[2]),args=(params[0],params[1],params[2]),dense_output=True)
    time=np.linspace(0,epoch*delta_t,epoch)
    return sequence.sol(time),time


def evaluation(start_index,end_index,f,prediction,reference):
    errors=[]
    for i in range(start_index,end_index):
        error=np.linalg.norm(torch.tensor(prediction[i])-torch.tensor(reference))/np.linalg.norm(torch.tensor(reference))
        errors.append(error)
    return errors

def plot_time_sequence(start_index,end_index,break_index,f,prediction,reference,time,dimensions):
    rows=dimensions
    errors = evaluation(start_index,end_index,f,prediction,reference)
    fig=plt.figure()
    fig.set_figwidth(40)
    fig.set_figheight(15)
    for i in range(rows):
        ax = fig.add_subplot(rows+1,1,i+1)
        ax.plot(time[start_index:end_index],reference.T[i][start_index:end_index])
        ax.plot(time[start_index:end_index],prediction.T[i][start_index:end_index])
        plt.axvline(x = time[break_index], color = 'b',linestyle='dashed')
    ax = fig.add_subplot(rows+1,1,rows+1)
    ax.plot(time[start_index:end_index],errors)
    plt.show()

def discrete_mackey_glass(params=(0.2,0.1,10,23,1000,.46,250),init=0,epoch=3000,delta_t=0.02):
    ### https://github.com/manu-mannattil/nolitsa/blob/master/nolitsa/data.py
    a,b,c,tau,n,sample,discard = params
    sample=int(n*sample/tau)

    grids = n * discard + sample * epoch
    x = np.empty(grids)

    x[:n] = 0.5 + 0.05 * (-1 + 2 * np.random.random(n))

    A = (2 * n - b * tau) / (2 * n + b * tau)
    B = a * tau / (2 * n + b * tau)

    for i in range(n - 1, grids - 1):
        x[i + 1] = A * x[i] + B * (x[i - n] / (1 + x[i - n] ** c) +
                                   x[i - n + 1] / (1 + x[i - n + 1] ** c))
        
    time=np.linspace(0,epoch*delta_t,epoch)
    sequence = []
    sequence.append(list(x[n*discard::sample]))
    return np.array(sequence),time