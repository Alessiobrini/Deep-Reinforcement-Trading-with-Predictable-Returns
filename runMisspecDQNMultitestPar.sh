#!/bin/sh
python runMisspecDQNPar.py
python -c 'import ruamel.yaml as yaml; from utils.common import save; from utils.common import format_tousands; f=open("config/paramMultiTestOOS.yaml"); y=yaml.safe_load(f); g=open("config/paramMisspecDQN.yaml"); x=yaml.safe_load(g); y["outputClass"]= x["outputClass"]; y["outputModel"]=x["outputModel"];  y["length"]= format_tousands(x["N_train"]); save(y,"config")'
python runMultiTestOOSParbySeed.py