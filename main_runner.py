import pdb
import gin
import argparse
import os
import sys
import time
import itertools
from itertools import combinations
from joblib import Parallel, delayed
import numpy as np
import ruamel.yaml as yaml


from runners.runDQN import RunDQNTraders
from runners.runMisspecDQN import RunMisspecDQNTraders

from runners.runPPO import RunPPOTraders
from runners.runMisspecPPO import RunMisspecPPOTraders

from runners.runDDPG import RunDDPGTraders

from utils.common import readConfigYaml
from utils.common import chunks
from utils.common import save
from utils.common import format_tousands

# TODO integrate these two scripts into a single one
from runners.runMultiTestOOSParbySeed import runMultiTestOOSbySeed
from runners.runPPOMultiTestOOSParbySeed import runPPOMultiTestOOSbySeed


def RunMultiParallelExp(var_par: list, Param: dict, func: object):
    """Parallelize the runner passed as function

    Parameters
    ----------
    var_par: list
        List of varying parameters for the single parallelized experiment

    Param: dict
        The dictionary containing the parameters

    func: object
        Runner to parallelize
    """

    for i in range(len(var_par)):
        Param[Param["varying_pars"][i]] = var_par[i]

    func(Param)


@gin.configurable()
def main_runner(configs_path: str, algo: str, experiment: str, parallel: bool):
    """Main function to run both a single experiment or a
    set of parallelized experiment

    Parameters
    ----------
    configs_path: str
        Path where the config files are stored

    algo: str
        Acronym of the algorithm to run. Read the comments in the gin config to see
        the available algorithms

    experiment: str
        Name of the type of synthetic experiment to perform. Read the comments in the gin config to see
        the available algorithms

    parallel: bool
        Choose to parallelize or not the selected experiments
    """
    func = None
    # get runner to do the experiments
    if algo == "DQN":

        if experiment == "GP":
            func = RunDQNTraders
            configname = "paramDQN.yaml"

        elif experiment == "Misspec":
            func = RunMisspecDQNTraders
            configname = "paramMisspecDQN.yaml"

    elif algo == "PPO":

        if experiment == "GP":
            func = RunPPOTraders
            configname = "paramPPO.yaml"

        elif experiment == "Misspec":
            func = RunMisspecPPOTraders
            configname = "paramMisspecPPO.yaml"

    elif algo == "DDPG":

        if experiment == "GP":
            func = RunDDPGTraders
            configname = "paramDDPG.yaml"

        elif experiment == "Misspec":
            # TODO add DDPG with dynamic misspecifications
            pass
    if func is None:
        print("Choose a proper algorithm and experiment type!")
        sys.exit()
    Param = readConfigYaml(os.path.join(configs_path, configname))

    # launch runner (either parallelized or not)
    if parallel:

        # select all the combinations of hyperparameters
        if (Param["varying_type"] == "combination") or (
            Param["varying_type"] == "chunk"
        ):
            variables = [
                xs
                for xs in itertools.product(*[Param[v] for v in Param["varying_pars"]])
            ]

        elif Param["varying_type"] == "ordered_combination":
            variables = [xs for xs in zip(*[Param[v] for v in Param["varying_pars"]])]

        elif Param["varying_type"] == "random_search":
            variables = [
                xs
                for xs in itertools.product(*[Param[v] for v in Param["varying_pars"]])
            ]
            variables = [
                variables[i]
                for i in np.random.randint(
                    0, len(variables) - 1, Param["num_rnd_search"]
                )
            ]

        else:
            print("Choose proper way to combine varying parameters")
            sys.exit()

        num_cores = len(variables)

        # run the combinations in parallel
        if Param["varying_type"] == "random_search":
            Parallel(n_jobs=num_cores)(
                delayed(RunMultiParallelExp)(var_par, Param) for var_par in variables
            )
            time.sleep(10)
            os.execv(sys.executable, ["python"] + sys.argv)

        if Param["varying_type"] == "chunk":
            num_cores = Param["num_rnd_search"]
            for chunk_var in chunks(variables, num_cores):
                Parallel(n_jobs=num_cores)(
                    delayed(RunMultiParallelExp)(var_par, Param, func)
                    for var_par in chunk_var
                )
                time.sleep(10)

        else:
            Parallel(n_jobs=num_cores)(
                delayed(RunMultiParallelExp)(var_par, Param, func)
                for var_par in variables
            )

    else:

        func(Param)

    # transfer path of the current experiment among yaml files
    f = open(os.path.join(configs_path, "paramMultiTestOOS.yaml"))
    paramtest = yaml.safe_load(f)

    g = open(os.path.join(configs_path, configname))
    x = yaml.safe_load(g)

    paramtest["outputClass"] = x["outputClass"]
    paramtest["outputModel"] = x["outputModel"]
    paramtest["length"] = format_tousands(x["N_train"])

    # run OOS tests
    # TODO integrate PPO out-of-sample in the original testing function
    if algo != "PPO":
        runMultiTestOOSbySeed(p=paramtest)
    else:
        runPPOMultiTestOOSbySeed(p=paramtest)


if __name__ == "__main__":
    example_text = """Examples of use:
    python main_runner.py --config main_config.gin
    """

    parser = argparse.ArgumentParser(
        description="DRL model runner.",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", type=str, dest="config", help="specify config file name"
    )

    args = parser.parse_args()

    configs_path = os.path.join(os.getcwd(), "config")

    # parse gin config file
    if args.config:
        gin_path = os.path.join(configs_path, args.config)
    else:
        gin_path = os.path.join(configs_path, "main_config.gin")
    gin.parse_config_file(gin_path, skip_unknown=True)

    main_runner(configs_path=configs_path)
