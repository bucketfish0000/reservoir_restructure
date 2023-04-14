import utils
import numpy as np
import json
import torch
import os
from collections import OrderedDict

class reservoirModel:
    def __init__(self,config_path):
        np.random.seed(1024)
        global config
        with open(config_path) as config_file:
            config = json.load(config_file,object_pairs_hook=OrderedDict)
        
        self.run_time = config["system"]["length"]
        self.training_time =  config["params"]["train time"]
        self.d_m=config["system"]["dimension"]
        self.d_r=config["reservoir"]["dimension"]
        self.d_t = config["system"]["d_t"]
        self.system=eval(config["system"]["function"])

        #eval(self.system+"()")
        #self.system=utils.integration_lorenz

        ### generate system input ###
        self.system_input,self.time = self.system(epoch=self.run_time,delta_t=self.d_t)
        self.system_input=torch.tensor(self.system_input.T)
        ### initialize input layer ###
        sigma = config["params"]["sigma"]
        self.W_in=self.init_in_layer(sigma)

        ### initialize reservoir weights ###
        roh = config["params"]["roh"]
        reservoir_degree=config["reservoir"]["degree"]
        self.W_reservoir=self.init_reservoir(roh,reservoir_degree)

        ### initialize reservoir states ###
        self.states=[] # list of tensors; states[i] is state vect of reservoir at time=i
        self.states.append(torch.tensor(np.random.rand(self.d_r))) # use a random state as start

        ### initialize output layer ###
        self.W_out=self.initOutLayer()

        ### initialize output list ###
        self.output=[]

        ### regression param ###
        self.lamda = config["params"]["lamda"]
        
        ### eval param ###
        self.f=config["params"]["f"]
        

    def training(self):
        ### initialize ###
        self.states,self.output=self.states[0:1],[]
        for i in range(self.training_time):
            ### at each time feed the corresponding input through ###
            reservoir_state,output = self.forward(i,torch.reshape(self.states[-1],[-1]))
            ### record everything
            self.states.append(torch.reshape(reservoir_state,[-1]))
            self.output.append(torch.reshape(output,[-1]))
        ### do ridge regression ###
        # convert everything to tensors
        recorded_states = torch.stack(self.states[1:])
        recorded_outputs = torch.stack(self.output)
        # we are making predictions so it is needed to 
        # compute the regression of outputs generated from inputs[t_0=0,t_e=e]
        # against the references from [t_0=1,t_e=e+1]
        reference_outputs = torch.tensor(self.system_input[1:self.training_time+1]) 
        #print(len(reference_outputs))
        # beta_hat = (X^T*X+lamda*I)^-1*(X^T*Y)fig2=plt.figure()
        # here X is state(i.e.input to output layer), Y is reference (system gen. output, i.e. target output from output layer)

        self.W_out = torch.tensor(np.dot(np.linalg.inv(torch.tensor(np.dot(recorded_states.T,recorded_states))+torch.tensor(self.lamda*(np.eye(self.d_r)))),np.dot(recorded_states.T,reference_outputs)))
        '''
        print("X^T*X:\n",np.dot(recorded_states.T,recorded_states))
        print("lamda*I:\n",self.lamda*(np.eye(self.d_r)))
        print("(X^T*X+lamda*I)^-1:\n",np.linalg.inv(torch.tensor(np.dot(recorded_states.T,recorded_states))+torch.tensor(self.lamda*(np.eye(self.d_r)))))
        print("X^T*Y:\n",np.dot(recorded_states.T,reference_outputs))
        '''
        
    def forward(self,epoch,prev):
        #print(len(prev))
        ### get corresponding input ###
        input = torch.tensor(self.system_input[epoch])
        ### filter thru input layer ###
        feed_to_reservoir = torch.tensor(np.dot(input,self.W_in))
        ### update reservoir state ###
        state = torch.tensor(torch.tensor(np.tanh(torch.tensor(np.dot(self.W_reservoir,prev))+torch.tensor(feed_to_reservoir))))
        ### compute output ###
        output = torch.tensor(np.dot(state,self.W_out))
        #print(state)
        ### return updated reservoir state and output ###
        return state,output
    
    def auto_forward(self,prev):
        ### get corresponding input ###
        input = torch.tensor(self.output[-1])
        ### filter thru input layer ###
        feed_to_reservoir = torch.tensor(np.dot(input,self.W_in))
        ### update reservoir state ###
        state = torch.tensor(torch.tensor(np.tanh(torch.tensor(np.dot(self.W_reservoir,prev))+torch.tensor(feed_to_reservoir))))
        ### compute output ###
        output = torch.tensor(np.dot(state,self.W_out))
        ### return updated reservoir state and output ###
        return state,output

    def run_by_self(self):
        # runs from the very beginning of all inputs/datasets
        #prediction_output = []print
        run_states = []
        run_states.append(self.states[-1])
        for i in range(self.training_time,self.run_time):
            state,prediction = self.auto_forward(torch.reshape(run_states[-1],[-1]))
            self.output.append(torch.reshape(prediction,[-1]))
            run_states.append(torch.reshape(state,[-1]))
        return torch.stack(self.output),run_states
    
    def run_with_input(self):
        run_states = []
        run_states.append(self.states[-1])
        prediction_output = []
        for i in range(self.run_time):
            state,prediction = self.forward(i,torch.reshape(run_states[-1],[-1]))
            prediction_output.append(torch.reshape(prediction,[-1]))
            run_states.append(torch.reshape(state,[-1]))
        return torch.stack(prediction_output),run_states
    

    def init_in_layer(self,sigma):
        return torch.tensor(np.random.uniform(-sigma,sigma,(self.d_m,self.d_r)))
    
    def init_reservoir(self,roh,reservoir_degree):
        W_reservoir = np.zeros((self.d_r,self.d_r))
        for node in W_reservoir:
            # random degree for each node around <d>
            number_connections=round(max(0,np.random.normal(reservoir_degree,reservoir_degree/3,1)[0]))
            # make random connections according to weight
            connections = np.random.choice(np.linspace(0,self.d_r-1,self.d_r,dtype=np.int),number_connections,replace=False)
            # random connection weight
            for connection in connections:
                node[connection]=np.random.uniform(-1,1)
        
        # re-normalize W_reservoir to roh
        max_eigen=max(np.abs(np.linalg.eig(W_reservoir)[0]))
        W_reservoir=(roh/max_eigen)*(torch.tensor(W_reservoir))
        return W_reservoir
    
    def initOutLayer(self):
        return torch.tensor(np.random.rand(self.d_r,self.d_m))        
            

    

