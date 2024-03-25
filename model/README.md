# Model instructions

Here you can find the instructions for running the simulations of the model.

## Content

### Scripts for running the standard version of the model

- [structural_plasticity.h](structural_plasticity.h): header file of structural_plasticity.cpp
- [structural_plasticity.cpp](structural_plasticity.cpp): code file, with all the functions to run every aspect of the simulation
- [test_structural_plasticity.cpp](test_structural_plasticity.cpp): code file that includes the previous codes and performs the simulation
- [params.dat](params.dat): contains all the simulation parameters. For an extensive description of the parameters please see the section below
- [make_structural_plasticity.sh](make_structural_plasticity.sh): script for compiling the model. Returns the object code ```test_structural_plasticity```
- [run.sh](run.sh): bash script for running sets of simulations using different seeds in supercomputers with SLURM job scheduler
- [run_sbatch.templ](run_sbatch.templ): template script for running the model in supercomputers with SLURM job scheduler

### Parameters of the model

| **Parameter** | **Description** | **Values** | **Default** |
|---|---|---:|---:|
| connection_rule | Type of connection rule | 0: fixed indegree, 1: poisson indegree, 2: fixed total number, 3: poisson total number | 1 |
| lognormal_rate | Neurons rate distribution | 1: lognormal distribution, 0: discrete rate  | 1 |
| allow_multapses | Allows multiple connections between two neurons. N.B. it works only with fixed indegree connection rule | 0: false, 1: true | 1 |
| r | Rewiring step | int (0: rewiring disabled) | 100 |
| T | Number of training patterns | int | 1000 |
| n_test | Number of test patterns to be provided to the network | int | 1000 |
| C | Number of connections per neuron of population 2 (i.e., indegree) | int | 5000 |
| alpha1 | probability for a neuron of population 1 to be at high rate regime for a pattern | float between 0 and 1 | 1e-3 |
| alpha2 | probability for a neuron of population 2 to be at high rate regime for a pattern | float between 0 and 1 | 1e-3 |
| N1 | Number of neurons in population 1 | int | 100000 |
| N2 | Number of neurons in population 2 | int | 100000 |
| Wb | Baseline synaptic weight [pA] | float | 0.1 |
| Ws | Stabilized and potentiated synaptic weight [pA] | float | 1.0 |
| nu_l_1 | Average rate for low-rate regime of neurons of population 1 [Hz] | float | 2.0 |
| nu_h_1 | Average rate for high-rate regime of neurons of population 1 [Hz] | float | 50.0 |
| nu_l_2 | Average rate for low-rate regime of neurons of population 2 [Hz] | float | 2.0 |
| nu_h_2 | Average rate for high-rate regime of neurons of population 2 [Hz] | float | 50.0 |
| noise_flag | Add noise when providing patterns in test phase | 0: false, 1: true | 1 |
| rate_noise | Standard deviation of the noise distribution (i.e., a truncated normal distribution) | float | 1.0 |
| max_noise_dev | Value that sets where to truncate the noise distribution (i.e., at +/- rate_noise*max_noise_dev) | float | 2.0 |
| corr_neg_rate | Correct the neurons firing rate when noise is applied and the firing rate reaches negative values | 0: no change, 1: truncated (extract again the noise value until the rate is non negative), 2: saturated (set automatically the rate to 0 Hz) | 0 |
| master_seed | Seed for random number generation | int | 123456 |
| load_network | Load network from a file | 0: false, 1: true | 0 |
| train_flag | Performs the training process | 0: false, 1: true | 1 |
| test_flag | Performs the test process | 0: false, 1: true | 1 |
| save_network | Saves the network to a file after training | 0: false, 1: true | 0 |
| random_test_order | Order of patterns given during test | 0: sequential, 1: random | 1 |
| train_block_size | number of traning patterns to be done before saving the network to file and update simulation status | int | 5000 |
| test_block_size | number of traning patterns to be done before updating simulation status | int | 5000 |

### Additional scripts

- [erfinv.cpp](erfinv.cpp): code that computes the inverse of the error function
- [background.cpp](background.cpp): code that performs the estimation of the bias related to possible input correlation between the neurons of population 2. It computes the unbiased estimation of $\sigma^2_b$
- [make_bkg.sh](make_bkg.sh): bash script for compiling background.cpp. Returns the object code ```background```


