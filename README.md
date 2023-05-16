Run `reservoir.ipynb` to train and auto-run the RP model as well as make a plot of outputs in time. See more in the file.

TODO:

    0.5. testcases
    
    3. mackey-glass fix

    - aim(python)

    -learning curve & lyapunov
    x
    
    -qualities


----

form of configs: (expired)

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

----

qualification of network:

-Lyapunov: sensitivity of trained model auto-run trajectory on initial conditions
    <https://www.mdpi.com/2311-5521/6/10/348>
    <https://hypertextbook.com/chaos/lyapunov-1/>

-Learning Curve: sensitivity of model output on training size; loss vs. % of sequence trained
 
-KR: lin. separability of resevoir output. rank of readout/subsample sequence from reservoir in response to finite random input sequence
multi-measure and average/smoothened

-GR: 

Theorem 5.1. Let r be the rank of the n × s matrix consisting
of the s vectors xu (t0 ) for all inputs u in S_univ (we assume that
S_univ is finite and contains s inputs). Then r ≤ VC-dimension
(HC ) ≤ r + 1.

We propose to use the rank r defined in Theorem 5.1 as
an estimate of VC-dimension (HC ), and hence as a measure
that informs us about the generalization capability of a neural
microcircuit C (for arbitrary probability distributions over the
set S_univ ).

    <https://arxiv.org/pdf/1810.07135.pdf>
    <https://www.sciencedirect.com/science/article/pii/S0893608007000433?via%3Dihub>

-MC: 