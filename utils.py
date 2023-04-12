import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def discreteLorenz(params=(10,28,8/3),init=(25,25,25), epoch=3000,delta_t=0.01,dimension=3):
    result = []
    result.append(init)
    for i in range(0,epoch):
        prev = result[i-1]
        curr = [prev[0]+delta_t*(params[0]*(prev[1]-prev[0])),
                prev[1]+delta_t*(prev[0]*(params[1]-prev[2])-prev[1]),
                prev[2]+delta_t*(prev[0]*prev[1]-params[2]*prev[2])]
        result.append(curr)
    return result

def integrationLorenz(params=(10,28,8/3),init=(25,25,25), epoch=300,delta_t=0.1,dimension=3):
    lorenz=solve_ivp(dynamicLorenz,(0,epoch),(init[0],init[1],init[2]),args=(params[0],params[1],params[2]),dense_output=True)
    time=np.linspace(0,epoch*delta_t,epoch)
    #x,y,z=lorenz.sol(time)
    return lorenz.sol(time),time

def dynamicLorenz(t,init,a,b,c):
    x,y,z=init
    dx=a*(y-x)
    dy=(b-z)*x-y
    dz=-c*z+x*y
    return dx,dy,dz


######## WASTED ###########
def plot(values,time,dimension=3):
    t = 0
    
    for seq in values:
        data = []
        for i in range(dimension):
            data.append([])   
        for vect in seq:
            for i in range(dimension):
                data[i].append(vect[i])
        plt.subplot(311)
        plt.plot(time,data[0])
        plt.subplot(312)
        plt.plot(time,data[1])
        plt.subplot(313)
        plt.plot(time,data[2])
    
    plt.show()