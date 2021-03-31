# Deep Reinforcement Trading with Predictable Returns

This code repository comes with the submitted paper to the IJCAI2021. It contains the main scripts and the utilities that allow to run the experiments and produce the figures attached to the paper. 

## Requirements

**Installation** -  Use the file `pip_requirements.txt` to install all the packages in a Linux virtual environment. The same holds for a conda environment or if you use a Windows OS.

A prerequisite is **Python 3.6** or greater.

Create an environment:

`python3 -m venv DRTrading`

Activate the environment:

`source DRTrading/bin/activate`

Install the required packages inside the environment:

`python3 -m pip install -r pip_requirements.txt`

## Source Code Structure

In the main folder there is a script called `main_runner.py` which represents the main script to launch all the simulations though runners, which are located in the subfolder with the same name. Each runners can let the user run an algorithm (DQN,PPO or DDPG) over simulated synthetic data of different type. The subfolder `utils` contains all classes and functions used to run the experiments and they are imported in the main scripts and runners.

## Usage

All the `*.py` files have their own documentation, both for class methods and functions. There are two main kind of experiments that the user can perform by running the following scripts:
- `runDQN.py` with associated config file for parameter called `paramDQN.yaml`. This script performs a DQN training on a return series driven by Gaussian mean-reverting factors.
- `runMisspecDQN.py` with associated config file for parameter called `paramMisspecDQN.yaml`. This script performs a DQN training on a return series simulated by a model misspecified as for the cases presented in the paper.

The same holds for the other two available algorithms (PPO and DDPG). **TODO: DDPP still needs the misspecified implementation**

Note that both script in the case of DQN allows also to run *Q-learning* in parallel with *DQN* algorithm. All the parameters for the simulation and the hyperparameters for the algorithm are passed to the scripts through their associated `*.yaml` file. Every config file contains also relevant information to choose the hyperparameters and all the general settings for the experiments. Config files are stored in the folder of the same name.

For an extensive analysis, the scripts can be run many time in parallel on different cores of your machine. This allows to do either extensive grid searches for hyperparameter tuning or to perform robustness checks of the algorithm by running on many different seeds at a time.


The configuration of the experiment type and its parallelization is done with `gin-config` ([github repo](https://github.com/google/gin-config)), a configuration framework for Python built by a team in Google. From the project description:

"[gin is] based on dependency injection. Functions or classes can be decorated with `@gin.configurable`, allowing default parameter values to be supplied from a config file (or passed via the command line) using a simple but powerful syntax. This removes the need to define and maintain configuration objects (e.g. protos), or write boilerplate parameter plumbing and factory code, while often dramatically expanding a project's flexibility and configurability."

If you have never worked with this framework, before moving on, read the README.md in the above linked repo. Our config files are stored in `/quant/configs` and you can create a different config for each experiment.

The user can visualize the evolution of some diagnostics during the DQN training as the gradient norms, intermediate layer outputs and losses with tensorboard just by running the command `tensorboard --logdir exp_directory` where `exp_directory` is the output folder for the corresponding experiment or set of experiments. Then the user can open the URL http://localhost:6006 on its machine. In this way also many different experiments can be visualized together, if the `exp_directory` folder contains them.

The repository contains a runner called `runMultiTestOOSPArbySeed.py` that allows the user to perform all the needed out-of-sample tests and to produce the same figures of the paper. 

Note that a third config file called `paramMultiTestOOS.yaml` regulates some hyperparameters for performing the out-of-sample tests and also for plotting the same figures of the paper. The script for plotting is `runAggMultiTestPlots.py`, which reproduces different kinds of plots according to the flags passed from the `paramMultiTestOOS.yaml` config file. 

### Randomness
As the user can see from the config files, randomness is a key ingredient of these experiments and therefore it is controlled by some hyperparameters. In general randomness arises in two situations:
- The simulation of the synthetic return series
- The initialization and the training of the DRL algorithm

The experiments of the type `runDQN.py` allows also to decouple the two sources of stochasticity by passing two different seed numbers. However, in our proposed experiment, we let those two sources be driven by the same `RandomState`.

# Endnotes for a better usage
The repository can be used in many different ways. 

One can run a simple experiment on a simulated financial series, one can run a grid search parallelizing over many different hyperparameter combinations just by passing the name of the varying parameters in the specific config file, or one can run the same hyperparameter combination over many different seeds.

The results proposed in our paper mostly do the last option. Thanks to the parameters passed through  the config files, the experiments are generally stored in a path like "outputDir\outputClass\outputModel\length\single_experiment_name".

Understanding this folder tree structure is very important when you want to use the attached code and go through the results of such extensive simulations. In general, each "single_experiment_name" has a certain number of associated checkpoints stored inside that folder. The `*MultiTestOOS*.py` scripts are able to take every checkpoints and perform many different tests out-of-sample for each, in order to do the above mentioned robustness check. Please note that those scripts need to know the name "outputModel" and the "length" of the folder where checkpoints are stored. More details are provided in `paramMultiTestOOS.yaml`. 

The script for plotting figures is not general enough to produce whatever the user wants, but it is specifically customized to produce multiple subplots to be attached to the paper and the supplementary material. However it is simple enough, so that it can be modified with a low effort, if one wants to do a different amount of subplots.

# Next step and TODOs

This repository should be intended as a work in progress and represents a first step in understanding the reasoning of DRL algorithms when they have to interpret and anayze financial trading signals.

From the point of view of the code, it is effective it can be improved in its efficiency and readability. A modular approach should be adopted in many different part of the code and the `gin` style of passing hyperparameters should be entirely adopted instead of `yaml` files which are easy to use but it causes redundancy in the code.
