# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:34:03 2020

@author: aless
"""

import os
if any('SPYDER' in name for name in os.environ):
    from IPython import get_ipython
    get_ipython().magic('reset -sf')

import os, logging, nbformat, sys, subprocess, time
from nbconvert.preprocessors import ExecutePreprocessor
from utils.generateLogger import generate_logger

start = time.time()

#create logger
logger = generate_logger()

# function to walk the dir tree only one level down
def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

# pass the name of the notebook and the folder of experiments
notebook_filename = os.path.join(os.getcwd(),"Analyze DQN results.ipynb")

# one can provide different set of experiments by passing this list
experiments_folders = ["outputs/DDQN_20200427/Decays/2M"]
# experiments_folder_path = os.path.join(os.getcwd(),experiments_folder)

# loop over the experiments provided
for fld in experiments_folders:
    logging.info('Folder {} starting'.format(fld))
    experiments_folder_path = os.path.join(os.getcwd(),fld)
    # loop over the folders of experiment
    for subdir, _, _ in walklevel(experiments_folder_path): #subdir,dirs,files
        
        # walklevel select also the current folder with the number of iteration, which is ok because we need 
        # to pass also that number to the nb
        if len(os.path.split(subdir)[-1]) < 5:
            os.environ['N_TRAIN'] = os.path.split(subdir)[-1]
        
        if len(os.path.split(subdir)[-1]) > 5:
            os.environ['FOLDER'] = subdir
            
            # read and execute notebook
            logging.info('Reading notebook...')
            with open(notebook_filename) as f:
                nb = nbformat.read(f, as_version=4)
                logging.info('Executing notebook...')
                ep = ExecutePreprocessor(timeout=600)#, kernel_name='python3')
                ep.preprocess(nb, {'metadata': {'path': os.getcwd()}})
                
            # save executed notebook in the proper folder
            with open(os.path.join(os.environ['FOLDER'],'ExecNbResult_{}.ipynb'.format(os.path.split(subdir)[-1])), 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
                logging.info('Saved executed notebook.')
            
            # save the executed notebook in html format
            subprocess.call(['jupyter', 
                             'nbconvert', 
                             '{}'.format(os.path.join(os.environ['FOLDER'],'ExecNbResult_{}.ipynb'.format(os.path.split(subdir)[-1]))),
                             '--output-dir', 
                             '{}'.format(os.environ['FOLDER'])])
                
            logging.info('Experiment {} has ended'.format(os.path.split(subdir)[-1]))
         

end = time.time()
logging.info('Script has finshed in {} minutes'.format((end - start)/60))
