Run `reservoir.ipynb` to train and auto-run the RP model as well as make a plot of outputs in time. See more in the file.

TODO:

    0.5. testcases
    
    3. mackey-glass fix

    - aim(python)

    -learning curve & lyapunov
    x
    
    -qualities


----

form of configs:

    {
        "system": {
            "dimension": <int> dimension of system 
            "length": <int>,
            "d_t": <float>,
            "function": <string> func name and call path to generate time-system
        },
        "reservoir": {
            "dimension": <int>,
            "degree": <int> avg/expected degree,
            "subsample": <int> # of subsampled dimensions (None for no subsampling)
            "network": <string> func name and call path to generate r network (None for default random)
        },
        "params": {
            "sigma": <int> input layer entry range (-sigma,sigma), 
            "rho": <float> reservoir spectral radius (0<rho<1),
            "train time": <int>,
            "lamda": <float> regression I-factor,
            "bias": <float> regression bias,
            "f": <float> evaluation error standard
        }
    }
