#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 16:48:05 2021


This script measures the following for the 8 ML models.

– Forward precision for different ML model studied. The foreward precision is 
    defined as the proportion of the firms with probs at top 1 percentile with 
    an AAER case identified by SEC after the study year over the number of firms 
    with with the probs at top percentile.

– Average number of years an alarm is made in advanced of SEC identifying a firm
    with an AAER.

uses 11 financial ratios to predict the likelihood of fraud in 
a financial statement. The financial ratios are based on Dechow et al (2011).
You access the Dechow paper at https://doi.org/10.1111/j.1911-3846.2010.01041.x

Inputs: 
    – A csv file under name "FraudDB2020.csv" with information regarding 
    US companies and AAER status. The column "AAER_DUMMY" is the identifier of the
    firms with a fraud case. 
    Expected columns are: 	
    0	fyear, 1	gvkey, 2	sich, 3	indfmt,4	cik, 5	datadate,6	fic,7	AAER_OVER,
    8	AAER_DUMMY, 9	act, 10	ap, 11	at, 12	ceq, 13	che, 14	cogs, 15	csho,
    16	dlc, 17	dltis, 18	dltt, 19	dp, 20	ib, 21	invt, 22	ivao, 23	ivst,
    24	lct, 25	lt, 26	ni, 27	ppegt, 28	pstk, 29	re, 30	rect, 31	sale,
    32	sstk, 33	txc, 34	txp, 35	xint, 36	prcc_f, 37	dwc, 38	rsst, 39	dreceivables,
    40	dinventory, 41	soft_assets, 42	chcs, 43	chcashmargin, 44	droa,
    45	dfcf, 46	issuance, 47	bm, 48	depindex, 49	ebitat, 50	reat
    – A pickle file including the information for SVM with Financial Kernel.


Predictive models:
    – Support Vector Machine with Financial Kernel as in Cecchini et al 2010;
        this model uses 23 raw variables (SVM-FK)
    – RUSBoost as Bao et al 2020; this model uses 28 raw variables (RUSBOOST)
    – Support Vector Machine with a linear kernel; this model uses 11 financial 
        ratios (SVM)
    – Logistic Regression; this model uses 11 financial ratios (KR)
    – SGD Tree Boosting; this model uses 11 financial ratios (SGD)
    – LogitBoost; this model uses 11 financial ratios (ADA)
    – MUlti-layered Perceptron; this model uses 11 financial ratios (MLP)
    – FUSED (weighted average of estimated probs of other methods); this model 
        uses 11 financial ratios

Outputs: 
Main results are stored in the table variable "forward_tbl" written into
1 csv file for time period 2001-2010

Steps:
    1. Load the optimal hyperparameters from the previous steps
    2. Estimating the forward performance for each OOS period
    3. Calculate the average, std, min, and max.

Warnings: 
    – This code requires a pickle file from running the SVM FK script. 
    – Running this code can take up to 120 mins if the Financial Kernel pickle 
    file is already loaded.
    These figures are estimates based on a MacBook Pro 2020.

@author: Arman Hassanniakalager GitHub: https://github.com/hkalager
Common disclaimers apply. Subject to change at all time.

Last review: 19/05/2022
"""
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from imblearn.ensemble import RUSBoostClassifier
from datetime import datetime
import pickle
import os
from statsmodels.stats.weightstats import ttest_ind
import warnings
warnings.filterwarnings("ignore")

# start the clock!
t0=datetime.now()
# setting the parameters
IS_period=10
k_fold=10
OOS_period=1
start_OOS_year=2001
end_OOS_year=2010
adjust_serial=True
fraud_df=pd.read_csv('FraudDB2020.csv')

fyears_available=np.unique(fraud_df.fyear)
count_over=count_fraud=np.zeros(fyears_available.shape)

reduced_tbl_1=fraud_df.iloc[:,[0,1,3,7,8]]
reduced_tbl_2=fraud_df.iloc[:,9:-3]
reduced_tblset=[reduced_tbl_1,reduced_tbl_2]
reduced_tbl=pd.concat(reduced_tblset,axis=1)
reduced_tbl=reduced_tbl[reduced_tbl.fyear>=1991]

tbl_fk_svm=reduced_tbl.copy()

tbl_fk_svm.pop('act')
tbl_fk_svm.pop('ap')
tbl_fk_svm.pop('ppegt')
tbl_fk_svm.pop('dltis')
tbl_fk_svm.pop('sstk')

# cross-validation optimised parameters used below for the ratio-based ML models

n_opt=1000
r_opt=1e-4
score_rus28=0.6505273864153011

opt_params_svm={'class_weight': {0: 0.01, 1: 1}, 'kernel': 'linear'}
C_opt=opt_params_svm['class_weight'][0]
kernel_opt=opt_params_svm['kernel']
score_svm=0.701939025416111

opt_params_lr={'class_weight': {0: 0.05, 1: 1}}
C_opt_lr=opt_params_lr['class_weight'][0]
score_lr=0.701876350738009

opt_params_sgd={'class_weight': {0: 5e-3, 1: 1}, 'loss': 'log', 'penalty': 'l2'}
score_sgd=0.7026775920776185

opt_params_ada={'learning_rate': 0.9, 'n_estimators': 20}
score_ada=0.700229450411913

opt_params={'activation': 'logistic', 'hidden_layer_sizes': 5, 'solver': 'adam'}
score_mlp=0.706333862286029

 
# Optimised setting for Cecchini et al (2010) – SVM Kernel
opt_params_svm_fk={'class_weight':{0: 0.02, 1: 1}}
score_svm=0.595534973722555
if os.path.isfile('features_fk.pkl')==True:    
    dict_db=pickle.load(open('features_fk.pkl','r+b'))
    tbl_ratio_fk=dict_db['lagged_Data']
    mapped_X=dict_db['matrix']
    red_tbl_fk=tbl_ratio_fk.iloc[:,-46:]
    print('pickle file for SVM-FK loaded successfully ...')
else:    
    raise NameError ('The pickle file for the financial kernel missing. Rerun the SVM FK script first...')


range_oos=range(start_OOS_year,end_OOS_year+1,OOS_period)

precision_rus1=np.zeros(len(range_oos))
avg_early_rus1=np.zeros(len(range_oos))
med_early_rus1=np.zeros(len(range_oos))

precision_svm_fk1=np.zeros(len(range_oos))
avg_early_svm_fk1=np.zeros(len(range_oos))
med_early_svm_fk1=np.zeros(len(range_oos))

precision_svm1=np.zeros(len(range_oos))
avg_early_svm1=np.zeros(len(range_oos))
med_early_svm1=np.zeros(len(range_oos))

precision_lr1=np.zeros(len(range_oos))
avg_early_lr1=np.zeros(len(range_oos))
med_early_lr1=np.zeros(len(range_oos))

precision_sgd1=np.zeros(len(range_oos))
avg_early_sgd1=np.zeros(len(range_oos))
med_early_sgd1=np.zeros(len(range_oos))

precision_ada1=np.zeros(len(range_oos))
avg_early_ada1=np.zeros(len(range_oos))
med_early_ada1=np.zeros(len(range_oos))

precision_mlp1=np.zeros(len(range_oos))
avg_early_mlp1=np.zeros(len(range_oos))
med_early_mlp1=np.zeros(len(range_oos))

precision_fused1=np.zeros(len(range_oos))
avg_early_fused1=np.zeros(len(range_oos))
med_early_fused1=np.zeros(len(range_oos))

m=0
for yr in range_oos:
    t1=datetime.now()
    year_start_IS=1991

    # Setting the IS for all models but SVM-FK
    tbl_year_IS=reduced_tbl.loc[np.logical_and(reduced_tbl.fyear<yr,\
                                               reduced_tbl.fyear>=year_start_IS)]
    tbl_year_IS=tbl_year_IS.reset_index(drop=True)
    misstate_firms=np.unique(tbl_year_IS.gvkey[tbl_year_IS.AAER_DUMMY==1])
    tbl_year_OOS=reduced_tbl.loc[np.logical_and(reduced_tbl.fyear>=yr,\
                                                reduced_tbl.fyear<yr+OOS_period)]
    X=tbl_year_IS.iloc[:,-11:]
    X_rus=tbl_year_IS.iloc[:,-39:-11]
    mean_vals=np.mean(X)
    std_vals=np.std(X)
    X=(X-mean_vals)/std_vals
    Y=tbl_year_IS.AAER_DUMMY
    
    # Setting the IS for SVM-FK
    idx_IS_FK=tbl_ratio_fk[np.logical_and(tbl_ratio_fk.fyear<yr,\
                                               tbl_ratio_fk.fyear>=year_start_IS)].index
    
    X_FK=mapped_X[idx_IS_FK,:]
    idx_real=np.where(np.logical_and(np.isnan(X_FK).any(axis=1)==False,\
                                     np.isinf(X_FK).any(axis=1)==False))[0]
    X_FK=X_FK[idx_real,:]
    X_FK=(X_FK-np.mean(X_FK,axis=0))/np.std(X_FK,axis=0)
    Y_FK=tbl_ratio_fk.AAER_DUMMY[idx_IS_FK]
    Y_FK=Y_FK.iloc[idx_real]    
    
    # Setting the OOS for all models but SVM-FK
    ok_index=np.zeros(tbl_year_OOS.shape[0])
    for s in range(0,tbl_year_OOS.shape[0]):
        if not tbl_year_OOS.iloc[s,1] in misstate_firms:
            ok_index[s]=True

    
    tbl_year_OOS=tbl_year_OOS.iloc[ok_index==True,:]
    tbl_year_OOS=tbl_year_OOS.reset_index(drop=True)
    
    X_OOS=tbl_year_OOS.iloc[:,-11:]
    X_OOS=(X_OOS-mean_vals)/std_vals
    
    X_rus_OOS=tbl_year_OOS.iloc[:,-39:-11]
    
    Y_OOS=tbl_year_OOS.AAER_DUMMY
    
    
    # Setting the OOS for SVM-FK
    tbl_fk_OOS=tbl_ratio_fk.loc[np.logical_and(tbl_ratio_fk.fyear>=yr,\
                                                tbl_ratio_fk.fyear<yr+OOS_period)]
    
    ok_index_fk=np.zeros(tbl_fk_OOS.shape[0])
    for s in range(0,tbl_fk_OOS.shape[0]):
        if not tbl_fk_OOS.iloc[s,1] in misstate_firms:
            ok_index_fk[s]=True
            
    X_OOS_FK=mapped_X[tbl_fk_OOS.index,:]
    idx_real_OOS_FK=np.where(np.logical_and(np.isnan(X_OOS_FK).any(axis=1)==False,\
                                     np.isinf(X_OOS_FK).any(axis=1)==False))[0]
    X_OOS_FK=X_OOS_FK[idx_real_OOS_FK,:]
    X_OOS_FK=(X_OOS_FK-np.mean(X_FK,axis=0))/np.std(X_FK,axis=0)
    Y_OOS_FK=tbl_year_OOS.AAER_DUMMY
    Y_OOS_FK=Y_OOS_FK.iloc[idx_real_OOS_FK]
    Y_OOS_FK=Y_OOS_FK.reset_index(drop=True)
    tbl_fk_OOS=tbl_fk_OOS.reset_index(drop=True)
    
    tbl_forward=reduced_tbl.loc[reduced_tbl.fyear>=yr+OOS_period]
    
    ok_index_forward=np.zeros(tbl_forward.shape[0])
    for s in range(0,tbl_forward.shape[0]):
        if not tbl_forward.iloc[s,1] in misstate_firms:
            ok_index_forward[s]=True
        
    
    tbl_forward=tbl_forward.iloc[ok_index_forward==True,:]
    tbl_forward=tbl_forward.reset_index(drop=True)
    
    forward_misstatement=tbl_forward.loc[tbl_forward.AAER_DUMMY==1]
    forward_misstatement=forward_misstatement.reset_index(drop=True)
    
    forward_misstate_firms=np.unique(forward_misstatement['gvkey'])
    
    # SVM with FK as in Cecchini et al 2010
    
    clf_svm_fk=SVC(class_weight=opt_params_svm_fk['class_weight'],kernel='linear',shrinking=False,\
                    probability=False,random_state=0,cache_size=1000,\
                        tol=X_FK.shape[-1]*1e-3)
    clf_svm_fk=clf_svm_fk.fit(X_FK,Y_FK)
    pred_test_svm_fk=clf_svm_fk.decision_function(X_OOS_FK)
    pred_test_svm_fk[pred_test_svm_fk>=1]=1+np.log(pred_test_svm_fk[pred_test_svm_fk>=1])
    probs_oos_fraud_svm_fk=np.exp(pred_test_svm_fk)/(1+np.exp(pred_test_svm_fk))
    
    labels_svm_fk=clf_svm_fk.predict(X_OOS_FK)
    
    cutoff_OOS_svm_fk=np.percentile(probs_oos_fraud_svm_fk,99)
    
    idx_top_1fk=np.where(probs_oos_fraud_svm_fk>=cutoff_OOS_svm_fk)[0]
    firms_top1fk=tbl_fk_OOS['gvkey'][idx_top_1fk].values
    forward_alarm_svm_fk1=0
    when_detect=[]
    for frm in firms_top1fk:
        if frm in forward_misstate_firms:
            forward_alarm_svm_fk1+=1
            idx_detect=np.where(forward_misstatement['gvkey']==frm)[0]
            when_detect=forward_misstatement['fyear'][idx_detect]
    if len(when_detect)>0:
        avg_early_svm_fk1[m]=np.mean(when_detect-yr)
        med_early_svm_fk1=np.median(when_detect-yr)
    
    precision_svm_fk1[m]=forward_alarm_svm_fk1/len(idx_top_1fk)
    
    # RUSBoost of Bao et al 2020
    base_tree=DecisionTreeClassifier(min_samples_leaf=5)
    bao_RUSboost=RUSBoostClassifier(base_estimator=base_tree,n_estimators=n_opt,\
                     learning_rate=r_opt,sampling_strategy=1,random_state=0)
    clf_rusboost = bao_RUSboost.fit(X_rus,Y)
    probs_oos_fraud_rus=clf_rusboost.predict_proba(X_rus_OOS)[:,-1]
    
    labels_rusboost=clf_rusboost.predict(X_rus_OOS)
    
    cutoff_OOS_rus=np.percentile(probs_oos_fraud_rus,99)
    
    idx_top1rus=np.where(probs_oos_fraud_rus>=cutoff_OOS_rus)[0]
    firms_top1rus=tbl_year_OOS['gvkey'][idx_top1rus].values
    forward_alarm_rus1=0
    when_detect=[]
    for frm in firms_top1rus:
        if frm in forward_misstate_firms:
            forward_alarm_rus1+=1
            idx_detect=np.where(forward_misstatement['gvkey']==frm)[0]
            when_detect=forward_misstatement['fyear'][idx_detect]
    if len(when_detect)>0:
        avg_early_rus1[m]=np.mean(when_detect-yr)
        med_early_rus1[m]=np.median(when_detect-yr)
    
    precision_rus1[m]=forward_alarm_rus1/len(idx_top1rus)
    
    
    # Support Vector Machines
    
    clf_svm=SVC(class_weight={0:C_opt,1:1},kernel=kernel_opt,shrinking=False,\
                    probability=False,random_state=0,max_iter=-1,\
                        tol=X.shape[-1]*1e-3)
        
    clf_svm=clf_svm.fit(X,Y)
    
    pred_test_svm=clf_svm.decision_function(X_OOS)
    probs_oos_fraud_svm=np.exp(pred_test_svm)/(1+np.exp(pred_test_svm))
    

    
    labels_svm=clf_svm.predict(X_OOS)
    
    
    cutoff_OOS_svm=np.percentile(probs_oos_fraud_svm,99)
    idx_top1svm=np.where(probs_oos_fraud_svm>=cutoff_OOS_svm)[0]
    firms_top1svm=tbl_year_OOS['gvkey'][idx_top1svm].values
    forward_alarm_svm1=0
    when_detect=[]
    for frm in firms_top1svm:
        if frm in forward_misstate_firms:
            forward_alarm_svm1+=1
            idx_detect=np.where(forward_misstatement['gvkey']==frm)[0]
            when_detect=forward_misstatement['fyear'][idx_detect]
    if len(when_detect)>0:
        avg_early_svm1[m]=np.mean(when_detect-yr)
        med_early_svm1[m]=np.median(when_detect-yr)
    
    precision_svm1[m]=forward_alarm_svm1/len(idx_top1svm)
    
    
    # Logistic Regression – Dechow et al (2011)
    
    clf_lr = LogisticRegression(class_weight=opt_params_lr['class_weight'],\
                             random_state=None).fit(X,Y)

    probs_oos_lr=clf_lr.predict_proba(X_OOS)
    probs_oos_fraud_lr=probs_oos_lr[:,-1]

    labels_lr=clf_lr.predict(X_OOS)


    cutoff_OOS_lr=np.percentile(probs_oos_fraud_lr,99)
    
   
    idx_top1lr=np.where(probs_oos_fraud_lr>=cutoff_OOS_lr)[0]
    firms_top1lr=tbl_year_OOS['gvkey'][idx_top1lr].values
    forward_alarm_lr1=0
    when_detect=[]
    for frm in firms_top1lr:
        if frm in forward_misstate_firms:
            forward_alarm_lr1+=1
            idx_detect=np.where(forward_misstatement['gvkey']==frm)[0]
            when_detect=forward_misstatement['fyear'][idx_detect]
    
    if len(when_detect)>0:
        avg_early_lr1[m]=np.mean(when_detect-yr)
        med_early_lr1[m]=np.median(when_detect-yr)
    precision_lr1[m]=forward_alarm_lr1/len(idx_top1lr)
   
    
    # Stochastic Gradient Decent 

    clf_sgd=SGDClassifier(class_weight=opt_params_sgd['class_weight'],\
                          loss=opt_params_sgd['loss'], random_state=0,\
                           penalty=opt_params_sgd['penalty'],validation_fraction=.2,shuffle=False)
    clf_sgd=clf_sgd.fit(X,Y)
    probs_oos_fraud_sgd=clf_sgd.predict_proba(X_OOS)[:,-1]
    
    
    labels_sgd=clf_sgd.predict(X_OOS)
    
    cutoff_OOS_sgd=np.percentile(probs_oos_fraud_sgd,99)
    
    idx_top1sgd=np.where(probs_oos_fraud_sgd>=cutoff_OOS_sgd)[0]
    firms_top1sgd=tbl_year_OOS['gvkey'][idx_top1sgd].values
    forward_alarm_sgd1=0
    when_detect=[]
    for frm in firms_top1sgd:
        if frm in forward_misstate_firms:
            forward_alarm_sgd1+=1
            idx_detect=np.where(forward_misstatement['gvkey']==frm)[0]
            when_detect=forward_misstatement['fyear'][idx_detect]
    
    if len(when_detect)>0:
        avg_early_sgd1[m]=np.mean(when_detect-yr)
        med_early_sgd1[m]=np.median(when_detect-yr)
    precision_sgd1[m]=forward_alarm_sgd1/len(idx_top1sgd)
    
    
    # Adaptive Boosting with logistic regression for weak learners
    base_tree=LogisticRegression(random_state=0)
    clf_ada=AdaBoostClassifier(n_estimators=opt_params_ada['n_estimators'],\
                               learning_rate=opt_params_ada['learning_rate'],\
                                   base_estimator=base_tree,random_state=0)
    clf_ada=clf_ada.fit(X,Y)
    probs_oos_fraud_ada=clf_ada.predict_proba(X_OOS)[:,-1]
    
    
    labels_ada=clf_ada.predict(X_OOS)
    
    cutoff_OOS_ada=np.percentile(probs_oos_fraud_ada,99)
    
    idx_top1ada=np.where(probs_oos_fraud_ada>=cutoff_OOS_ada)[0]
    firms_top1ada=tbl_year_OOS['gvkey'][idx_top1ada].values
    forward_alarm_ada1=0
    when_detect=[]
    for frm in firms_top1ada:
        if frm in forward_misstate_firms:
            forward_alarm_ada1+=1
            idx_detect=np.where(forward_misstatement['gvkey']==frm)[0]
            when_detect=forward_misstatement['fyear'][idx_detect]
    
    if len(when_detect)>0:
        avg_early_ada1[m]=np.mean(when_detect-yr)
        med_early_ada1[m]=np.median(when_detect-yr)
    precision_ada1[m]=forward_alarm_ada1/len(idx_top1ada)

    
    # Multi Layer Perceptron
    clf_mlp=MLPClassifier(hidden_layer_sizes=opt_params['hidden_layer_sizes'], \
                          activation=opt_params['activation'],solver=opt_params['solver'],\
                                       random_state=0,validation_fraction=.1)
    clf_mlp=clf_mlp.fit(X,Y)
    probs_oos_fraud_mlp=clf_mlp.predict_proba(X_OOS)[:,-1]
    
    labels_mlp=clf_mlp.predict(X_OOS)
    
    cutoff_OOS_mlp=np.percentile(probs_oos_fraud_mlp,99)
    
    idx_top1mlp=np.where(probs_oos_fraud_mlp>=cutoff_OOS_mlp)[0]
    firms_top1mlp=tbl_year_OOS['gvkey'][idx_top1mlp].values
    forward_alarm_mlp1=0
    when_detect=[]
    for frm in firms_top1mlp:
        if frm in forward_misstate_firms:
            forward_alarm_mlp1+=1
            idx_detect=np.where(forward_misstatement['gvkey']==frm)[0]
            when_detect=forward_misstatement['fyear'][idx_detect]
    
    if len(when_detect)>0:
        avg_early_mlp1[m]=np.mean(when_detect-yr)
        med_early_mlp1[m]=np.median(when_detect-yr)
    precision_mlp1[m]=forward_alarm_mlp1/len(idx_top1mlp)
   
    
    # Fused approach
    
    weight_ser=np.array([score_svm,score_lr,score_sgd,score_ada,score_mlp])
    weight_ser=weight_ser/np.sum(weight_ser)
    
    clf_fused=np.dot(np.array([probs_oos_fraud_svm,\
                      probs_oos_fraud_lr,\
                          probs_oos_fraud_sgd,probs_oos_fraud_ada,\
                              probs_oos_fraud_mlp]).T,weight_ser)
    
    probs_oos_fraud_fused=clf_fused
    
    labels_fused=(clf_fused>=np.percentile(clf_fused,99)).astype(int)
    
    cutoff_OOS_fused=np.percentile(probs_oos_fraud_fused,99)
    
    idx_top1fused=np.where(probs_oos_fraud_fused>=cutoff_OOS_fused)[0]
    firms_top1fused=tbl_year_OOS['gvkey'][idx_top1fused].values
    forward_alarm_fused1=0
    when_detect=[]
    for frm in firms_top1fused:
        if frm in forward_misstate_firms:
            forward_alarm_fused1+=1
            idx_detect=np.where(forward_misstatement['gvkey']==frm)[0]
            when_detect=forward_misstatement['fyear'][idx_detect]
    
    if len(when_detect)>0:
        avg_early_fused1[m]=np.mean(when_detect-yr)
        med_early_fused1[m]=np.median(when_detect-yr)
    precision_fused1[m]=forward_alarm_fused1/len(idx_top1fused)
    
    t2=datetime.now() 
    dt=t2-t1
    print('analysis finished for OOS period '+str(yr)+' after '+str(dt.total_seconds())+' sec')
    m+=1

print('average top percentile sensitivity for the period '+str(start_OOS_year)+' to '+\
      str(end_OOS_year)+' is '+ str(round(np.mean(precision_svm_fk1)*100,2))+\
          '% for SVM-FK vs '+str(round(np.mean(precision_rus1)*100,2))+\
              '% for RUSBoost vs '+str(round(np.mean(precision_svm1)*100,2))+\
                  '% for SVM vs '+ str(round(np.mean(precision_lr1)*100,2))+\
                      '% for Dechow-LR vs '+ str(round(np.mean(precision_sgd1)*100,2))+\
                          '% for SGD vs '+ str(round(np.mean(precision_ada1)*100,2))+\
                              '% for ADA vs '+ str(round(np.mean(precision_mlp1)*100,2))+\
                                  '% for MLP vs '+ str(round(np.mean(precision_fused1)*100,2))+\
                                      '% for FUSED')

# create performance table now
forward_tbl=pd.DataFrame()
forward_tbl['models']=['SVM-FK-23','RUSBoost-28','SVM','LR','SGD','ADA','MLP','FUSED']


forward_tbl['Mean Forward Precision']=[np.mean(precision_svm_fk1),\
                                       np.mean(precision_rus1),np.mean(precision_svm1),\
                                 np.mean(precision_lr1),\
                                     np.mean(precision_sgd1),np.mean(precision_ada1),\
                                         np.mean(precision_mlp1),np.mean(precision_fused1)]

pval_svm_fk=ttest_ind(precision_svm1,precision_svm_fk1,alternative='smaller')[1]
pval_rus=ttest_ind(precision_svm1,precision_rus1,alternative='smaller')[1]
pval_svm=ttest_ind(precision_svm1,precision_svm1,alternative='smaller')[1]
pval_lr=ttest_ind(precision_svm1,precision_lr1,alternative='smaller')[1]
pval_sgd=ttest_ind(precision_svm1,precision_sgd1,alternative='smaller')[1]
pval_ada=ttest_ind(precision_svm1,precision_ada1,alternative='smaller')[1]
pval_mlp=ttest_ind(precision_svm1,precision_mlp1,alternative='smaller')[1]
pval_fused=ttest_ind(precision_svm1,precision_fused1,alternative='smaller')[1]
# SGD and LogitBoost are significant

pval_mw_svm_fk=mannwhitneyu(precision_svm1,precision_svm_fk1,alternative='less')[1]
pval_mw_rus=mannwhitneyu(precision_svm1,precision_rus1,alternative='less')[1]
pval_mw_svm=mannwhitneyu(precision_svm1,precision_svm1,alternative='less')[1]
pval_mw_lr=mannwhitneyu(precision_svm1,precision_lr1,alternative='less')[1]
pval_mw_sgd=mannwhitneyu(precision_svm1,precision_sgd1,alternative='less')[1]
pval_mw_ada=mannwhitneyu(precision_svm1,precision_ada1,alternative='less')[1]
pval_mw_mlp=mannwhitneyu(precision_svm1,precision_mlp1,alternative='less')[1]
pval_mw_fused=mannwhitneyu(precision_svm1,precision_fused1,alternative='less')[1]
# Only LogitBoost is significant


forward_tbl['ttest pval']=[pval_svm_fk,pval_rus,pval_svm,pval_lr,pval_sgd,\
                                 pval_ada,pval_mlp,pval_fused]

forward_tbl['Std Forward Precision']=[np.std(precision_svm_fk1),\
                                      np.std(precision_rus1),np.std(precision_svm1),\
                                 np.std(precision_lr1),\
                                     np.std(precision_sgd1),np.std(precision_ada1),\
                                         np.std(precision_mlp1),np.std(precision_fused1)]

forward_tbl['Median Forward Precision']=[np.median(precision_svm_fk1),\
                                       np.median(precision_rus1),np.median(precision_svm1),\
                                 np.median(precision_lr1),\
                                     np.median(precision_sgd1),np.median(precision_ada1),\
                                         np.median(precision_mlp1),np.median(precision_fused1)]    


# Measure the average number of years where a correct prediction is made 


forward_tbl['Mean Year Ahead']=[np.mean(avg_early_svm_fk1),np.mean(avg_early_rus1),\
                                np.mean(avg_early_svm1),np.mean(avg_early_lr1),\
                                     np.mean(avg_early_sgd1),np.mean(avg_early_ada1),\
                                         np.mean(avg_early_mlp1),np.mean(avg_early_fused1)]

pval_when_svm_fk=ttest_ind(avg_early_svm1,avg_early_svm_fk1,alternative='smaller')[1]
pval_when_rus=ttest_ind(avg_early_svm1,avg_early_rus1,alternative='smaller')[1]
pval_when_svm=ttest_ind(avg_early_svm1,avg_early_svm1,alternative='smaller')[1]
pval_when_lr=ttest_ind(avg_early_svm1,avg_early_lr1,alternative='smaller')[1]
pval_when_sgd=ttest_ind(avg_early_svm1,avg_early_sgd1,alternative='smaller')[1]
pval_when_ada=ttest_ind(avg_early_svm1,avg_early_ada1,alternative='smaller')[1]
pval_when_mlp=ttest_ind(avg_early_svm1,avg_early_mlp1,alternative='smaller')[1]
pval_when_fused=ttest_ind(avg_early_svm1,avg_early_fused1,alternative='smaller')[1]    
# None where significant

pval_mw_when_svm_fk=mannwhitneyu(avg_early_svm1,avg_early_svm_fk1,alternative='less')[1]
pval_mw_when_rus=mannwhitneyu(avg_early_svm1,avg_early_rus1,alternative='less')[1]
pval_mw_when_svm=mannwhitneyu(avg_early_svm1,avg_early_svm1,alternative='less')[1]
pval_mw_when_lr=mannwhitneyu(avg_early_svm1,avg_early_lr1,alternative='less')[1]
pval_mw_when_sgd=mannwhitneyu(avg_early_svm1,avg_early_sgd1,alternative='less')[1]
pval_mw_when_ada=mannwhitneyu(avg_early_svm1,avg_early_ada1,alternative='less')[1]
pval_mw_when_mlp=mannwhitneyu(avg_early_svm1,avg_early_mlp1,alternative='less')[1]
pval_mw_when_fused=mannwhitneyu(avg_early_svm1,avg_early_fused1,alternative='less')[1]   
# Only LogitBoost is significant


forward_tbl['Std Year Ahead']=[np.std(avg_early_svm_fk1),np.std(avg_early_rus1),\
                               np.std(avg_early_svm1),np.std(avg_early_lr1),\
                                     np.std(avg_early_sgd1),np.std(avg_early_ada1),\
                                         np.std(avg_early_mlp1),np.std(avg_early_fused1)]

forward_tbl['Median Year Ahead']=[np.median(avg_early_svm_fk1),np.median(avg_early_rus1),\
                                np.median(avg_early_svm1),np.median(avg_early_lr1),\
                                     np.median(avg_early_sgd1),np.median(avg_early_ada1),\
                                         np.median(avg_early_mlp1),np.median(avg_early_fused1)]

# forward_tbl['Min Year Ahead']=[np.min(avg_early_svm_fk1),np.min(avg_early_rus1),\
#                                np.min(avg_early_svm1),np.min(avg_early_lr1),\
#                                      np.min(avg_early_sgd1),np.min(avg_early_ada1),\
#                                          np.min(avg_early_mlp1),np.min(avg_early_fused1)]                                            

# forward_tbl['Max Year Ahead']=[np.max(avg_early_svm_fk1),np.max(avg_early_rus1),\
#                                np.max(avg_early_svm1), np.max(avg_early_lr1),\
#                                      np.max(avg_early_sgd1),np.max(avg_early_ada1),\
#                                          np.max(avg_early_mlp1),np.max(avg_early_fused1)]                                           

lbl_perf_tbl='forward_tbl_'+str(start_OOS_year)+'_'+str(end_OOS_year)+\
    ',OOS='+str(OOS_period)+',serial='+str(adjust_serial)+'.csv'


forward_tbl.to_csv(lbl_perf_tbl,index=False)
print(forward_tbl)
t_last=datetime.now()
dt_total=t_last-t0
print('total run time is '+str(dt_total.total_seconds())+' sec')
