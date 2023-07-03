# Deep Reinforcement Trading with Predictable Returns

This code repository comes with the paper [Deep reinforcement trading with predictable returns]([https://arxiv.org/abs/2104.14683](https://www.sciencedirect.com/science/article/abs/pii/S0378437123004569)) published on *Physica A: Statistical Mechanics and its Applications*. It contains the main scripts and the utilities that allow to run the experiments and produce the figures attached to the paper. 

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

In the main folder there is a script called `main_runner.py` which represents the main script to launch all the simulations through runners, that are located in the `runners` folder. Each runner can let the user run an algorithm (DQN and PPO) over simulated synthetic data of different types. The subfolder `utils` contains all classes and functions used to run the experiments and they are imported in the main scripts and runners.

## Usage

Almost all the `*.py` files have their own documentation, both for class methods and functions. There are two main kind of experiments that the user can perform by running the following scripts:
- A training on a return series driven by Gaussian mean-reverting factors.
- A training on a return series simulated by a model misspecified as for the cases presented in the paper.

All the parameters for the simulation and the hyperparameters for the algorithm are passed to the scripts through a `gin` file. This file contains also relevant information to choose the hyperparameters and all the general settings for the experiments. It is stored in the `config` folder.

For an extensive analysis, the scripts can be run many times in parallel on different cores of your machine. This allows to do either extensive grid searches for hyperparameter tuning or to perform robustness checks of the algorithm by running on many different seeds at a time.


The configuration of the experiment type and its parallelization is done with `gin-config` ([github repo](https://github.com/google/gin-config)), a configuration framework for Python built by a team in Google. From the project description:

"[gin is] based on dependency injection. Functions or classes can be decorated with `@gin.configurable`, allowing default parameter values to be supplied from a config file (or passed via the command line) using a simple but powerful syntax. This removes the need to define and maintain configuration objects (e.g. protos), or write boilerplate parameter plumbing and factory code, while often dramatically expanding a project's flexibility and configurability."

If you have never worked with this framework, before moving on, read the README.md in the above linked repo. Our gin files are stored in the `config` folder together with the `yaml` files.


The repository allows the user to perform all the needed out-of-sample tests and to produce the same figures of the paper. The parallelization “by seed” means that you run plenty of out-of-sample tests for each hyperparameter (or just seeds) combination at the same time. Please note that you machine can have a different amount of cores than the default settings, so change the amount of combinations you want to run at the same time in order to avoid memory problems.

Note that the config file called `paramMultiTestOOS.yaml` regulates some hyperparameters for performing the out-of-sample tests and also for plotting the same figures of the paper. The script for plotting is `plot_runner.py`, which reproduces different kinds of plots according to the flags passed from the `paramMultiTestOOS.yaml` config file.

### Randomness
As the user can see from the config files, randomness is a key ingredient of these experiments and therefore it is controlled by some hyperparameters. In general randomness arises in two situations:
- The simulation of the synthetic return series
- The initialization and the training of the DRL algorithm


# Endnotes for a better usage
The repository can be used in many different ways.

One can run a simple experiment on a simulated financial series, one can run a grid search parallelizing over many different hyperparameter combinations just by passing the name of the varying parameters in the specific config file, or one can run the same hyperparameter combination over many different seeds.

The results proposed in our paper mostly do the last option. Thanks to the parameters passed through  the config files, the experiments are generally stored in a path like `outputDir\outputClass\outputModel\length\single_experiment_name`.

Understanding this folder tree structure is very important when you want to use the attached code and go through the results of such extensive simulations. In general, each `single_experiment_name` has a certain number of associated checkpoints stored inside that folder. 

The script for plotting figures is not general enough to produce whatever the user wants, but it is specifically customized to produce multiple subplots to be attached to the paper and the supplementary material. However it is simple enough, so that it can be modified with a low effort, if one wants to do a different amount of subplots.

# Next steps and TODOs

This repository should be intended as a work in progress and represents a first step in understanding the reasoning of DRL algorithms when they have to interpret and analyze financial trading signals.

Further extensions will come as soon as possible.
