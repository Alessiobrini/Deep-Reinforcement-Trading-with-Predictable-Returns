import pdb
import gin
import argparse
import os
import sys
import time
from joblib import Parallel, delayed
from runners.DQN_runner import DQN_runner
from runners.PPO_runner import PPO_runner
from utils.common import chunks
from utils.parallel import get_parallelized_combinations

import warnings
warnings.filterwarnings("ignore")


def parallel_exps(var_par, varying_par_to_change, gin_path, func):
    """Main function to parallelize which loads the parameters for the real experiments
    and run both training and testing routines

    Parameters
    ----------
    var_par: list
        List of varying parameters for the single parallelized experiment

    Param: dict
        The dictionary containing the parameters
    """

    gin.parse_config_file(gin_path, skip_unknown=True)

    for i in range(len(var_par)):
        gin.bind_parameter(varying_par_to_change[i], var_par[i])

    model_runner = func()
    model_runner.run()


@gin.configurable()
def main_runner(configs_path: str, algo: str):
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
    
    # get runner to do the experiments
    if algo == "DQN":
        func = DQN_runner

    elif algo == "PPO":
        func = PPO_runner

    # launch runner (either parallelized or not)
    if gin.query_parameter("%VARYING_PARS") is not None:
        # get varying parameters, combinations and cores
        varying_type = gin.query_parameter("%VARYING_TYPE")
        varying_par_to_change = gin.query_parameter("%VARYING_PARS")
        combinations, num_cores = get_parallelized_combinations(varying_type)

        # choose way to parallelize
        if varying_type == "random_search":
            Parallel(n_jobs=num_cores)(
                delayed(parallel_exps)(
                    var_par, varying_par_to_change, gin_path, func=func
                )
                for var_par in combinations
            )
            time.sleep(5)
            os.execv(sys.executable, ["python"] + sys.argv)
        elif varying_type == "chunk":
            num_cores = gin.query_parameter("%NUM_CORES")
            for chunk_var in chunks(combinations, num_cores):
                Parallel(n_jobs=num_cores)(
                    delayed(parallel_exps)(
                        var_par, varying_par_to_change, gin_path, func=func
                    )
                    for var_par in chunk_var
                )
                time.sleep(5)
        else:
            print("Choose proper way to parallelize.")
            sys.exit()
    else:

        model_runner = func()
        model_runner.run()

if __name__ == "__main__":
    example_text = """Examples of use:
    python main_runner.py --config main_config.gin
    """

    parser = argparse.ArgumentParser(
        description="Model runner.",
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
        gin_path = os.path.join(configs_path, "single_asset_GP.gin") # "two_asset_GP.gin"
    gin.parse_config_file(gin_path, skip_unknown=True)

    main_runner(configs_path=configs_path)
