# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:15:06 2020

@author: aless
"""
from tqdm import tqdm
import os, pdb

def PreTraining(pt_returns, pt_factors, pt_f_speed, pt_env, PreTrainQNet, PreTargetQNet, N_pretrain, 
                epsilon, copy_step, savedpath, save_ckpt_pretrained_model, save_ckpt_pretrained_steps):


    CurrState,_ = pt_env.reset()
    iters = 0
    #epsilon = max(min_eps, epsilon - eps_decay) # linear decay
    # for now try only random actions
    for i in tqdm(iterable=range(N_pretrain + 1), desc='PreTraining DQNetwork'):
        shares_traded = PreTrainQNet.eps_greedy_action(CurrState, epsilon)
        NextState, Result, NextFactors = pt_env.step(CurrState, shares_traded, i)
        exp = {'s': CurrState, 'a': shares_traded, 'r': Result['Reward_DQN'], 's2': NextState, 'f': NextFactors}
        
        PreTrainQNet.add_experience(exp)
        PreTrainQNet.train(PreTargetQNet, i, pt_env)
          
        CurrState = NextState
        
        iters += 1
        if (iters % copy_step == 0) and (i > PreTrainQNet.start_train):
            PreTargetQNet.copy_weights(PreTrainQNet)
            
        if save_ckpt_pretrained_model and (i % save_ckpt_pretrained_steps == 0) and (i > 0):
            PreTrainQNet.model.save_weights(os.path.join(savedpath, 'ckpt_pt','DQN_{}_it_pretrained_weights'.format(i)), 
                                           save_format='tf')
    pdb.set_trace()
    PreTrainQNet.model.save_weights(os.path.join(savedpath,'DQN_pretrained_weights'), save_format='tf')
    
        
    