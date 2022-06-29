#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 13:38:08 2021

This script checks whether the necessary packages are imported in
Python and also includes two functions for NDCG and VIF.


The NDCG@k function is based on Bao et al (2020)
Original codes in Matlab are obtained from 
https://github.com/JarFraud/FraudDetection/blob/master/evaluate.m

@author: Arman Hassanniakalager GitHub: https://github.com/hkalager
Common disclaimers apply. Subject to change at all time.

Last review: 28/06/2022
"""

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
dot=np.dot
t=np.transpose
inv=np.linalg.inv
diag=np.diag
exp=np.exp


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

class relogit:
    
    __version__='1.0.0'
    def __init__(self, Y,X,add_const=False):
        '''
        
        Parameters
        ----------
        Y : array_like
            A 1-d endogenous response variable. See statsmodels guidance  
        X : array_like
            A nobs x k array where nobs is the number of observations and k is 
            the number of regressors. An intercept is added by setting add_const
            to True.
        add_const : Boolean, optional
            Whether to add a constant into X. The default is False.

        Returns
        -------
        None.

        '''
        if add_const is not False:
            X=add_constant(X)
        base_model=Logit(Y,X)
        fitted_model=base_model.fit(disp=0)
        params=fitted_model.params
        pred=fitted_model.predict()
        w=diag(pred*(1-pred))
        #Q <- X_matrix %*% solve(t(X_matrix) %*% W %*% X_matrix) %*% t(X_matrix)
        inner_1=inv(dot(dot(t(X),w),X))
        q=dot(dot(X,inner_1),t(X))
        #e <- 0.5 * diag(Q) * (2 * pred - 1)
        e=0.5*diag(q)*(2*pred-1)
        #bias <- (solve(t(X_matrix) %*% W %*% X_matrix) %*% t(X_matrix) %*% W %*% e)
        bias=dot(dot(dot(inv(dot(dot(t(X),w),X)),t(X)),w),e)
        self.base=base_model
        self.X=X
        self.Y=Y
        self.unbiased_params=params-bias
        self.unbiased_pred=base_model.predict(params=self.unbiased_params)
        self.a_c=add_const
    
    def predict(self,X_new=[]):
        if len(X_new)>0:
            X=X_new
            if self.a_c is True:
                X=add_constant(X)
            self.X=X
        else:
            X=self.X
        unbiased_params=self.unbiased_params
        unbiased_pred=(1+exp(-1*dot(X,unbiased_params)))**-1
        self.unbiased_pred=unbiased_pred
        return unbiased_pred
            