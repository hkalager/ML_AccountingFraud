#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 13:38:08 2021

This code is calculating NDCG@k based on Bao et al (2020)
Original codes in Matlab are obtained from 
https://github.com/JarFraud/FraudDetection/blob/master/evaluate.m
@author: arman
"""
import sys
import subprocess
import importlib

# Check for the necessary modules


numpy_spec = importlib.util.find_spec("numpy")
found_numpy = numpy_spec is not None

if found_numpy==False:
    _=subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                           'numpy'])
import numpy as np

pandas_spec = importlib.util.find_spec("pandas")
found_pandas = pandas_spec is not None

if found_pandas==False:
    _=subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                           'pandas'])
import pandas as pd

matplotlib_spec = importlib.util.find_spec("matplotlib")
found_matplotlib = matplotlib_spec is not None

if found_matplotlib==False:
    _=subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                           'matplotlib'])

sklearn_spec = importlib.util.find_spec("sklearn")
found_sklearn = sklearn_spec is not None

if found_sklearn==False:
    _=subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                           'sklearn'])


statsmodels_spec = importlib.util.find_spec("statsmodels")
found_statsmodels = statsmodels_spec is not None

if found_statsmodels==False:
    _=subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                           'statsmodels'])
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Define necessary modules



def ndcg_k(label_true,est_prob,prc=1,pos_class=1):
    #est_prob=np.round(est_prob,5)
    if label_true.shape[0]!=est_prob.shape[0]:
        raise ValueError('No data ... check the target series')
    # if fractional prc is provided multiple by 100
    if np.logical_and(type(prc)==float,prc<1):
        prc=prc*100
        print('fractional percentile detected ... check if this is intended ...')
    # if prc>50 change to 100-prc (top PRC percentile)
    if prc>50:
        prc=100-prc
    
    
    k=round(len(label_true)*prc/100)
    idx=np.argsort(-est_prob)
    hits = np.sum(label_true==pos_class)
    kz=min(k,hits)
    
    z=0
    for i in range(0,kz):
        rel = 1
        z = z+ (2^rel-1)/np.log2(2+i)

    dcg_at_k=0
    for i in range(0,k):
        if label_true[idx[i]]==1:
            rel = 1
            dcg_at_k = dcg_at_k + (2^rel-1)/np.log2(2+i)
    
    if z!=0:
        ndcg_at_k = dcg_at_k/z
    else:
        ndcg_at_k = 0

    return ndcg_at_k

# the evaluation function used to optimise labeling     
def eval_fn(cut_off,y_hat,y_target):
    lbl_predict=(y_hat>cut_off).astype(int)
     
    TPR=np.sum(np.logical_and(lbl_predict==1,y_target==1))/np.sum(y_target==1)
     
    TNR=np.sum(np.logical_and(lbl_predict==0,y_target==0))/np.sum(y_target==0)
    
    perf=-1*np.power(TPR*TNR,.5)
    return perf


def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return(vif)
