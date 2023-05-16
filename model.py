import os
import utils
import numpy as np
import torch
import copy

class ESNModel:
    def __init__(self, config, seed=None):
        np.random.seed(seed)  # FIXME: Use np.random.default_rng

        self.run_time = config["system"]["length"]
        self.training_time = int(config["params"]["train ratio"] * self.run_time)
        self.d_m = config["system"]["dimension"]
        self.d_r = config["reservoir"]["dimension"]
        self.d_t = config["system"]["d_t"]
        ### initialize input layer ###
        sigma = config["params"]["sigma"]
        self.W_in = self.init_in_layer(sigma)

        ### initialize reservoir weights ###
        rho = config["params"]["rho"]
        reservoir_sparsity = config["reservoir"]["sparsity"]
        self.network_function = self.random_init_reservoir
        self.W_reservoir = self.network_function(rho, reservoir_sparsity)
        self.subsample = int(config["reservoir"]["subsample"])
        if self.subsample == None:
            self.sample_size = self.d_r
        else:
            self.sample_size = 2 * self.subsample
        self.sample_entries = np.random.randint(
            low=0, high=self.d_r, size=self.subsample
        )

        self.leakage = config["reservoir"]["leakage"]
        self.time_const = config["reservoir"]["time const"]
        ### initialize reservoir states ###
        self.states = ([self.initialize_state])

        ### initialize output layer ###
        self.W_out = self.init_out_layer()

        ### regression param ###
        self.lamda = config["params"]["lamda"]

    def initialize_state(self):
        return torch.tensor(np.random.rand(self.d_r))

    def training(self, data, qualification=0):
        ### initialize ###
        training_data = data[:self.training_time]
        expected_data = data[1:self.training_time+1]
        subtrained = []
        input_count = 0


        observation_list = []
        prev_state = self.initialize_state()
        for inputs in training_data:
            input_count += 1
            ### at each time feed the corresponding input through ###
            prev_state, sample, _ = self.forward(
                prev_state, torch.tensor(inputs)
            )
            ### record 
            observation_list.append(sample)
            ### record subtrained for learning curve
            if qualification != 0 and input_count%(self.training_time/qualification):
                sub_ESN = copy.deepcopy(self)
                _,_ = sub_ESN.ridge_output_layer(observation_list,expected_data)
                subtrained.append(sub_ESN)

        loss,loss_before_training = self.ridge_output_layer(observation_list,expected_data)

        return loss, loss_before_training, subtrained

    def ridge_output_layer(self,observation_list,expected_data,):
        ### do ridge regression ###
        recorded_samples = torch.stack(observation_list)
        # we are making predictions so it is needed to
        # compute the regression of outputs generated from inputs[t_0=0,t_e=e]
        # against the references from [t_0=1,t_e=e+1]
        reference_outputs = torch.tensor(expected_data)
        # print(len(reference_outputs))
        # beta_hat = (X^T*X+lamda*I)^-1*(X^T*Y)
        # here X is state(i.e.input to output layer), Y is reference (system gen. output, i.e. target output from output layer)
        loss_before_training = np.linalg.norm(reference_outputs - recorded_samples @ self.W_out, axis=1).mean()
        self.W_out = torch.tensor(
            np.dot(
                np.linalg.inv(
                    torch.tensor(np.dot(recorded_samples.T, recorded_samples))
                    + torch.tensor(self.lamda * (np.eye(self.sample_size)))
                ),
                np.dot(recorded_samples.T, reference_outputs),
            )
        )
        loss = np.linalg.norm(reference_outputs - recorded_samples @ self.W_out, axis=1).mean()
        return loss, loss_before_training
    
    def forward(self, prev, input):
        # print(len(prev))
        ### get corresponding input ###
        # input = torch.tensor(self.system_input[epoch])
        ### filter thru input layer ###
        feed_to_reservoir = torch.tensor(np.dot(input, self.W_in))
        ### update reservoir state ###
        state = torch.tensor(
            torch.tensor(
                (1-self.leakage)*prev
            )
            +
            torch.tensor(
                self.leakage*
                np.tanh(
                    torch.tensor(np.dot(self.W_reservoir, prev))
                    + torch.tensor(feed_to_reservoir)
                )
            )
        )
        ### sample from reservoir ###
        sample = self.sample_from_reservoir(state)
        ### compute output ###
        output = torch.tensor(np.dot(sample, self.W_out))
        # print(state)
        ### return updated reservoir state and output ###
        return state, sample, output

    def run_by_self(self, data):
        # runs from the very beginning of all inputs/datasets
        # prediction_output = []
        output_list = []
        training_data = data[:self.training_time]

        # Load internal states to remember the past dynamics
        prev_state = self.initialize_state()
        for inputs in training_data:
            prev_state, sample, prediction = self.forward(
                prev_state, torch.tensor(inputs)
            )
            output = sample @ self.W_out
            output_list.append(output)

        # Predict based on generated past states
        for i in range(self.training_time, self.run_time):
            prev_state, sample, prediction = self.forward(
                prev_state, torch.tensor(output)
            )
            output = sample @ self.W_out
            output_list.append(output)

        return torch.stack(output_list)

    def init_in_layer(self, sigma):
        return torch.tensor(np.random.uniform(-sigma, sigma, (self.d_m, self.d_r)))

    def random_init_reservoir(self,rho,sparsity):
        W_reservoir = np.zeros((self.d_r, self.d_r))

        edge_number = round(sparsity * self.d_r**2)
        edge_row_indices=np.random.choice(np.linspace(0, self.d_r - 1, self.d_r, dtype=np.int),edge_number,replace=True)
        edge_column_indices=np.random.choice(np.linspace(0, self.d_r - 1, self.d_r, dtype=np.int),edge_number,replace=True)
        for row,column in zip(edge_row_indices,edge_column_indices):
            W_reservoir[row][column] = np.random.uniform(-1, 1)

        
        # re-normalize W_reservoir to rho
        max_eigen = max(np.abs(np.linalg.eig(W_reservoir)[0]))
        W_reservoir = (rho / max_eigen) * (torch.tensor(W_reservoir))
        return W_reservoir

    def init_out_layer(self):
        if self.subsample == None:
            return torch.tensor(np.random.rand(self.d_r, self.d_m))
        else:

            return torch.tensor(np.random.rand(2 * self.subsample, self.d_m))

    def sample_from_reservoir(self, state):
        # subsample from reservoir with non-lin augmentation
        # return whole reservoir state if no subsample
        if self.subsample == None:
            return state
        else:
            sample = []
            for i in range(self.subsample):
                sample.append(state[self.sample_entries[i]])
                sample.append(state[self.sample_entries[i]] ** 2)
            return torch.tensor(sample)
