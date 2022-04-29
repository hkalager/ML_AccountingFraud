#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 17:46:11 2021

This code uses 11 financial ratios to compare performance of an AdaBoost with 
decision tree learners with AdaBoost with Logistic Regression

Inputs: 
A csv file under name "FraudDB2020.csv" with information regarding 
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

Predictive models:
    – Adaptive Boosting with Decision Tree (AdaBoost-Tree)
    – Adaptive Boosting with Logistic Regression (LogitBoost)

Outputs: 
Main results are stored in the table variable "perf_tbl_general" written into
1 csv files: time period 2001-2010 

Steps:
    1. Cross-validate to find optimal hyperparameters.
    2. Estimating the performance for each OOS period.

Warnings: 
    – Running this code can take up to 510 mins. The cross-validation takes up
    to 450 mins (you can skip this step) main analysis up to 60 mins. 
    These figures are estimates based on a MacBook Pro 2017.

@author: Arman Hassanniakalager GitHub: https://github.com/hkalager
Common disclaimers apply. Subject to change at all time.

Last review: 29/04/2022
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
from extra_codes import ndcg_k

# start the clock!
t0=datetime.now()
# setting the parameters
IS_period=10
k_fold=10
OOS_period=1
start_OOS_year=2001
end_OOS_year=2010
C_FN=30 #relative to C_FP
C_FP=1 
adjust_serial=True
cross_val=False
case_window='expanding'
fraud_df=pd.read_csv('FraudDB2020.csv')


fyears_available=np.unique(fraud_df.fyear)
count_over=count_fraud=np.zeros(fyears_available.shape)

reduced_tbl_1=fraud_df.iloc[:,[0,1,3,7,8]]
reduced_tbl_2=fraud_df.iloc[:,-14:-3]
reduced_tblset=[reduced_tbl_1,reduced_tbl_2]
reduced_tbl=pd.concat(reduced_tblset,axis=1)
reduced_tbl=reduced_tbl[reduced_tbl.fyear>=1991]
reduced_tbl=reduced_tbl[reduced_tbl.fyear<=2010]

# Setting the cross-validation setting

tbl_year_IS_CV=reduced_tbl.loc[np.logical_and(reduced_tbl.fyear<2001,\
                                           reduced_tbl.fyear>=2001-IS_period)]
tbl_year_IS_CV=tbl_year_IS_CV.reset_index(drop=True)
misstate_firms=np.unique(tbl_year_IS_CV.gvkey[tbl_year_IS_CV.AAER_DUMMY==1])

X_CV=tbl_year_IS_CV.iloc[:,-11:]

mean_vals=np.mean(X_CV)
std_vals=np.std(X_CV)
X_CV=(X_CV-mean_vals)/std_vals

Y_CV=tbl_year_IS_CV.AAER_DUMMY

P_f=np.sum(Y_CV==1)/len(Y_CV)
P_nf=1-P_f

# redo cross-validation if you wish

if cross_val==True: 
    
    
    # optimise AdaBoost with logistic regression (Ada-LR)
    print('Grid search hyperparameter optimisation started for AdaBoost')
    t1=datetime.now()
    
    param_grid_ada={'n_estimators':[10,20,50,100,200,500],\
                    'learning_rate':[.1,5e-1,9e-1,1]}
        
    best_perf_ada=0
    class_weight_rng=[{0:2e-3,1:1},{0:5e-3,1:1},{0:1e-2,1:1},{0:2e-2,1:1},\
                      {0:5e-2,1:1},{0:1e-1,1:1},{0:2e-1,1:1},{0:5e-1,1:1},{0:1e0,1:1}]
    for class_ratio in class_weight_rng:
        base_lr=LogisticRegression(random_state=0,class_weight=class_ratio)
        base_mdl_ada=AdaBoostClassifier(base_estimator=base_lr,random_state=0)
        
        clf_ada_lr = GridSearchCV(base_mdl_ada, param_grid_ada,scoring='roc_auc',\
                           n_jobs=-1,cv=k_fold,refit=False)
        clf_ada_lr.fit(X_CV, Y_CV)
        score_ada_lr=clf_ada_lr.best_score_
        if score_ada_lr>=best_perf_ada:
            best_perf_ada=score_ada_lr
            opt_params_ada_lr=clf_ada_lr.best_params_
            opt_class_weight_ada_lr=class_ratio
        
    
    t2=datetime.now()
    dt=t2-t1
    print('AdaBoost-LR CV finished after '+str(dt.total_seconds())+' sec')
    
    print('AdaBoost-LR: The optimal number of estimators is '+\
          str(opt_params_ada_lr['n_estimators'])+', and learning rate '+\
              str(opt_params_ada_lr['learning_rate']))
    imbalance_fact=opt_class_weight_ada_lr[1]/opt_class_weight_ada_lr[0]
    print('AdaBoost-LR: The optimal C+/C- is '+str(imbalance_fact))
    
    # optimise AdaBoost with tree learners (Ada-Tree): this is the basic model    
    t1=datetime.now()
    
    best_perf_ada_tree=0
    for class_ratio in class_weight_rng:
        base_tree=DecisionTreeClassifier(min_samples_leaf=5,class_weight=class_ratio)
        base_mdl_ada=AdaBoostClassifier(base_estimator=base_tree,random_state=0)
        clf_ada_tree = GridSearchCV(base_mdl_ada, param_grid_ada,scoring='roc_auc',\
                       n_jobs=-1,cv=k_fold,refit=False)
        clf_ada_tree.fit(X_CV, Y_CV)
        score_ada_tree=clf_ada_tree.best_score_
        if score_ada_tree>best_perf_ada_tree:
            best_perf_ada_tree=score_ada_tree
            opt_params_ada_tree=clf_ada_tree.best_params_
            opt_class_weight_ada_tree=class_ratio
        
    
    t2=datetime.now()
    dt=t2-t1
    print('AdaBoost-Tree CV finished after '+str(dt.total_seconds())+' sec')
    
    print('AdaBoost-Tree: The optimal number of estimators is '+\
          str(opt_params_ada_tree['n_estimators'])+', and learning rate '+\
              str(opt_params_ada_tree['learning_rate']))
    
    imbalance_fact_tree=opt_class_weight_ada_tree[1]/opt_class_weight_ada_tree[0]
    print('AdaBoost-Tree: The optimal C+/C- is '+str(imbalance_fact_tree))
    print('Hyperparameter optimisation finished successfully.\nStarting the main analysis now...')
else:
    
    opt_params_ada_lr={'learning_rate': 0.9, 'n_estimators': 20}
    opt_class_weight_ada_lr={0:1e0,1:1}
    score_ada_lr=0.700229450411913
    
    opt_params_ada_tree={'learning_rate': 0.1, 'n_estimators': 500}
    opt_class_weight_ada_tree={0: 0.02, 1: 1}
    score_ada_tree=0.6584007795597742


range_oos=range(start_OOS_year,end_OOS_year+1,OOS_period)


roc_ada_tree=np.zeros(len(range_oos))
specificity_ada_tree=np.zeros(len(range_oos))
sensitivity_OOS_ada_tree=np.zeros(len(range_oos))
precision_ada_tree=np.zeros(len(range_oos))
sensitivity_OOS_ada_tree1=np.zeros(len(range_oos))
specificity_OOS_ada_tree1=np.zeros(len(range_oos))
precision_ada_tree1=np.zeros(len(range_oos))
ndcg_ada_tree1=np.zeros(len(range_oos))
ecm_ada_tree1=np.zeros(len(range_oos))


roc_ada_lr=np.zeros(len(range_oos))
specificity_ada_lr=np.zeros(len(range_oos))
sensitivity_OOS_ada_lr=np.zeros(len(range_oos))
precision_ada_lr=np.zeros(len(range_oos))
sensitivity_OOS_ada_lr1=np.zeros(len(range_oos))
specificity_OOS_ada_lr1=np.zeros(len(range_oos))
precision_ada_lr1=np.zeros(len(range_oos))
ndcg_ada_lr1=np.zeros(len(range_oos))
ecm_ada_lr1=np.zeros(len(range_oos))



m=0
for yr in range_oos:
    t1=datetime.now()
    if case_window=='expanding':
        year_start_IS=1991
    else:
        year_start_IS=yr-IS_period
    
    tbl_year_IS=reduced_tbl.loc[np.logical_and(reduced_tbl.fyear<yr,\
                                               reduced_tbl.fyear>=year_start_IS)]
    tbl_year_IS=tbl_year_IS.reset_index(drop=True)
    misstate_firms=np.unique(tbl_year_IS.gvkey[tbl_year_IS.AAER_DUMMY==1])
    tbl_year_OOS=reduced_tbl.loc[np.logical_and(reduced_tbl.fyear>=yr,\
                                                reduced_tbl.fyear<yr+OOS_period)]
    
    if adjust_serial==True:
        ok_index=np.zeros(tbl_year_OOS.shape[0])
        for s in range(0,tbl_year_OOS.shape[0]):
            if not tbl_year_OOS.iloc[s,1] in misstate_firms:
                ok_index[s]=True
            
        
    else:
        ok_index=np.ones(tbl_year_OOS.shape[0]).astype(bool)
        
    
    tbl_year_OOS=tbl_year_OOS.iloc[ok_index==True,:]
    tbl_year_OOS=tbl_year_OOS.reset_index(drop=True)
        
    
    X=tbl_year_IS.iloc[:,-11:]
    mean_vals=np.mean(X)
    std_vals=np.std(X)
    X=(X-mean_vals)/std_vals
    Y=tbl_year_IS.AAER_DUMMY
    
    X_OOS=tbl_year_OOS.iloc[:,-11:]
    X_OOS=(X_OOS-mean_vals)/std_vals
    
    Y_OOS=tbl_year_OOS.AAER_DUMMY
    
    n_P=np.sum(Y_OOS==1)
    n_N=np.sum(Y_OOS==0)
    
    # Adaptive Boosting with logistic regression for weak learners (LogitBoost)
    base_lr=LogisticRegression(random_state=0,class_weight=opt_class_weight_ada_lr)
    clf_ada_lr=AdaBoostClassifier(n_estimators=opt_params_ada_lr['n_estimators'],\
                               learning_rate=opt_params_ada_lr['learning_rate'],\
                                   base_estimator=base_lr,random_state=0)
    clf_ada_lr=clf_ada_lr.fit(X,Y)
    probs_oos_fraud_ada_lr=clf_ada_lr.predict_proba(X_OOS)[:,-1]
    
    
    labels_ada_lr=clf_ada_lr.predict(X_OOS)
    
    roc_ada_lr[m]=roc_auc_score(Y_OOS,probs_oos_fraud_ada_lr)
    specificity_ada_lr[m]=np.sum(np.logical_and(labels_ada_lr==0,Y_OOS==0))/\
        np.sum(Y_OOS==0)
    if np.sum(labels_ada_lr)>0:
        sensitivity_OOS_ada_lr[m]=np.sum(np.logical_and(labels_ada_lr==1, \
                                                 Y_OOS==1))/np.sum(Y_OOS)
        precision_ada_lr[m]=np.sum(np.logical_and(labels_ada_lr==1,Y_OOS==1))/np.sum(labels_ada_lr)
    
    cutoff_OOS_ada_lr=np.percentile(probs_oos_fraud_ada_lr,99)
    sensitivity_OOS_ada_lr1[m]=np.sum(np.logical_and(probs_oos_fraud_ada_lr>=cutoff_OOS_ada_lr, \
                                                 Y_OOS==1))/np.sum(Y_OOS)
    specificity_OOS_ada_lr1[m]=np.sum(np.logical_and(probs_oos_fraud_ada_lr<cutoff_OOS_ada_lr, \
                                                  Y_OOS==0))/np.sum(Y_OOS==0)
    precision_ada_lr1[m]=np.sum(np.logical_and(probs_oos_fraud_ada_lr>=cutoff_OOS_ada_lr, \
                                                 Y_OOS==1))/np.sum(probs_oos_fraud_ada_lr>=cutoff_OOS_ada_lr)
    ndcg_ada_lr1[m]=ndcg_k(Y_OOS,probs_oos_fraud_ada_lr,99)
    
    FN_ada_lr1=np.sum(np.logical_and(probs_oos_fraud_ada_lr<cutoff_OOS_ada_lr, \
                                                  Y_OOS==1))
    FP_ada_lr1=np.sum(np.logical_and(probs_oos_fraud_ada_lr>=cutoff_OOS_ada_lr, \
                                                  Y_OOS==0))
        
    ecm_ada_lr1[m]=C_FN*P_f*FN_ada_lr1/n_P+C_FP*P_nf*FP_ada_lr1/n_N
        
    
    
    
    # Adaptive Boosting with decision trees as weak learners (AdaBoost)
    base_tree=DecisionTreeClassifier(min_samples_leaf=5,class_weight=opt_class_weight_ada_tree)
    clf_ada_tree=AdaBoostClassifier(n_estimators=opt_params_ada_tree['n_estimators'],\
                               learning_rate=opt_params_ada_tree['learning_rate'],\
                                   base_estimator=base_tree,random_state=0)
    clf_ada_tree=clf_ada_tree.fit(X,Y)
    probs_oos_fraud_ada_tree=clf_ada_tree.predict_proba(X_OOS)[:,-1]
    
    
    labels_ada_tree=clf_ada_tree.predict(X_OOS)
    
    roc_ada_tree[m]=roc_auc_score(Y_OOS,probs_oos_fraud_ada_tree)
    specificity_ada_tree[m]=np.sum(np.logical_and(labels_ada_tree==0,Y_OOS==0))/\
        np.sum(Y_OOS==0)
    if np.sum(labels_ada_tree)>0:
        sensitivity_OOS_ada_tree[m]=np.sum(np.logical_and(labels_ada_tree==1, \
                                                 Y_OOS==1))/np.sum(Y_OOS)
        precision_ada_tree[m]=np.sum(np.logical_and(labels_ada_tree==1,Y_OOS==1))/np.sum(labels_ada_tree)
    
    cutoff_OOS_ada_tree=np.percentile(probs_oos_fraud_ada_tree,99)
    sensitivity_OOS_ada_tree1[m]=np.sum(np.logical_and(probs_oos_fraud_ada_tree>=cutoff_OOS_ada_tree, \
                                                 Y_OOS==1))/np.sum(Y_OOS)
    specificity_OOS_ada_tree1[m]=np.sum(np.logical_and(probs_oos_fraud_ada_tree<cutoff_OOS_ada_tree, \
                                                  Y_OOS==0))/np.sum(Y_OOS==0)
    precision_ada_tree1[m]=np.sum(np.logical_and(probs_oos_fraud_ada_tree>=cutoff_OOS_ada_tree, \
                                                 Y_OOS==1))/np.sum(probs_oos_fraud_ada_tree>=cutoff_OOS_ada_tree)
    ndcg_ada_tree1[m]=ndcg_k(Y_OOS,probs_oos_fraud_ada_tree,99)
    
    FN_ada_tree1=np.sum(np.logical_and(probs_oos_fraud_ada_tree<cutoff_OOS_ada_tree, \
                                                  Y_OOS==1))
    FP_ada_tree1=np.sum(np.logical_and(probs_oos_fraud_ada_tree>=cutoff_OOS_ada_tree, \
                                                  Y_OOS==0))
        
    ecm_ada_tree1[m]=C_FN*P_f*FN_ada_tree1/n_P+C_FP*P_nf*FP_ada_tree1/n_N
        
    
    
    t2=datetime.now() 
    dt=t2-t1
    print('analysis finished for OOS period '+str(yr)+' after '+str(dt.total_seconds())+' sec')
    m+=1

print('average top percentile sensitivity for the period '+str(start_OOS_year)+' to '+\
      str(end_OOS_year)+' is '+ str(round(np.mean(sensitivity_OOS_ada_lr1)*100,2))+\
                          '% for LogitBoost vs '+ str(round(np.mean(sensitivity_OOS_ada_tree1)*100,2))+\
                              '% for AdaBoost')


# create performance table now
perf_tbl_general=pd.DataFrame()
perf_tbl_general['models']=['AdaBoost','LogitBoost']
perf_tbl_general['Roc']=[np.mean(roc_ada_tree),np.mean(roc_ada_lr)]

                                            
perf_tbl_general['Sensitivity @ 1 Prc']=[np.mean(sensitivity_OOS_ada_tree1),\
                                 np.mean(sensitivity_OOS_ada_lr1)]

perf_tbl_general['Specificity @ 1 Prc']=[np.mean(specificity_OOS_ada_tree1),\
                                 np.mean(specificity_OOS_ada_lr1)]

perf_tbl_general['Precision @ 1 Prc']=[np.mean(precision_ada_tree1),\
                                 np.mean(precision_ada_lr1)]

perf_tbl_general['F1 Score @ 1 Prc']=2*(perf_tbl_general['Precision @ 1 Prc']*\
                                      perf_tbl_general['Sensitivity @ 1 Prc'])/\
                                        ((perf_tbl_general['Precision @ 1 Prc']+\
                                          perf_tbl_general['Sensitivity @ 1 Prc']))
perf_tbl_general['NDCG @ 1 Prc']=[np.mean(ndcg_ada_tree1),\
                                 np.mean(ndcg_ada_lr1)]   

perf_tbl_general['ECM @ 1 Prc']=[np.mean(ecm_ada_tree1),\
                                 np.mean(ecm_ada_lr1)]
    
                                          
lbl_perf_tbl='Compare_Ada'+str(start_OOS_year)+'_'+str(end_OOS_year)+\
    '_'+case_window+',OOS='+str(OOS_period)+','+\
    str(k_fold)+'fold'+',serial='+str(adjust_serial)+'_11ratios.csv'


perf_tbl_general.to_csv(lbl_perf_tbl,index=False)
print(perf_tbl_general)
t_last=datetime.now()
dt_total=t_last-t0
print('total run time is '+str(dt_total.total_seconds())+' sec')
