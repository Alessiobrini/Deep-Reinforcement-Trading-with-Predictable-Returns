# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:15:06 2020

@author: aless
"""
from tqdm import tqdm
import os

def PreTraining(pt_returns, pt_factors, pt_f_speed, pt_env, PreTrainNet, PreTargetNet, N_pretrain, 
                epsilon, copy_step, savedpath, save_ckpt_pretrained_model):


    CurrState = pt_env.reset()
    iters = 0
    #epsilon = max(min_eps, epsilon - eps_decay) # linear decay
    # for now try only random actions
    for i in tqdm(iterable=range(N_pretrain + 1), desc='PreTraining DQNetwork'): 
        shares_traded = PreTrainNet.eps_greedy_action(CurrState, epsilon)
        NextState, Result, NextFactors = pt_env.step(CurrState, shares_traded, i)
        exp = {'s': CurrState, 'a': shares_traded, 'r': Result['Reward_DQN'], 's2': NextState, 'f': NextFactors}
        
        PreTrainNet.add_experience(exp)
        PreTrainNet.train(PreTargetNet, i, pt_env)
          
        CurrState = NextState
        
        iters += 1
        if (iters % copy_step == 0) and (i > PreTrainNet.start_train):
            PreTargetNet.copy_weights(PreTrainNet)
            
        if save_ckpt_pretrained_model and (i % save_ckpt_pretrained_model == 0):
            PreTrainNet.model.save_weights(os.path.join(savedpath, 'ckpt_pt','DQN_{}_it_pretrained_weights'.format(i)), 
                                           save_format='tf')

    PreTrainNet.model.save_weights(os.path.join(savedpath,'DQN_pretrained_weights'), save_format='tf')
        
        
    