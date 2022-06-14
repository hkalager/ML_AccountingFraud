#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:10:53 2021

This code uses 11 financial ratios to predict the likelihood of fraud in 
a financial statement. The financial ratios are based on Dechow et al (2011).
You access the Dechow paper at https://doi.org/10.1111/j.1911-3846.2010.01041.x

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
    – Support Vector Machine (SVM)
    – Logistic Regression (LR)
    – SGD Tree Boosting (SGD)
    – Adaptive Boosting with Logistic Regression/LogitBoost (ADA)
    – MUlti-layered Perceptron (MLP)
    – FUSED (weighted average of estimated probs of other methods)

Outputs: 
Main results are stored in the table variable "perf_tbl_general" written into
2 csv files: time period 2001-2010 and 2003-2008. 

Steps:
    1. Cross-validate to find optimal hyperparameters.
    2. Estimating the performance for each OOS period.

Warnings: 
    – Running this code can take up to 85 mins. The cross-validation takes up
    to 60 mins (you can skip this step) main analysis up to 25 mins. 
    These figures are estimates based on a MacBook Pro 2021.

@author: Arman Hassanniakalager GitHub: https://github.com/hkalager
Common disclaimers apply. Subject to change at all time.

Last review: 19/05/2022
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
from extra_codes import ndcg_k
import warnings
warnings.filterwarnings("ignore")
# start the clock!
t0=datetime.now()
# setting the parameters
IS_period=10
k_fold=10
OOS_period=1 # 1 year ahead prediction
OOS_gap=0 # Gap between training and testing period
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

print('prior probablity of fraud between 1991-2000 is '+str(np.round(P_f*100,2))+'%')

# redo cross-validation if you wish

if cross_val==True: 
    # optimize RUSBoost grid
    print('Grid search hyperparameter optimisation started for RUSBoost')
    t1=datetime.now()
    param_grid_rusboost={'n_estimators':[10,20,50,100,200,500,1000],
                         'learning_rate':[1e-4,1e-3,1e-2,.1]}
    base_tree=DecisionTreeClassifier(min_samples_leaf=5)
    bao_RUSboost=RUSBoostClassifier(base_estimator=base_tree,\
                     sampling_strategy=1,random_state=0)
    clf_rus = GridSearchCV(bao_RUSboost, param_grid_rusboost,scoring='roc_auc',\
                       n_jobs=-1,cv=k_fold,refit=False)
    clf_rus.fit(X_CV, Y_CV)
    opt_params_rus=clf_rus.best_params_
    n_opt=opt_params_rus['n_estimators']
    r_opt=opt_params_rus['learning_rate']
    score_rus=clf_rus.best_score_
        
    t2=datetime.now()
    dt=t2-t1
    print('RUSBoost CV finished after '+str(dt.total_seconds())+' sec')
    print('RUSBoost: The optimal number of estimators is '+str(n_opt))
    
    # optimize SVM grid
    
    print('Grid search hyperparameter optimisation started for SVM')
    t1=datetime.now()
    param_grid_svm={'kernel':['linear','rbf','poly'],'class_weight':[\
                                    {0:2e-3,1:1},{0:5e-3,1:1},
                                    {0:1e-2,1:1},{0:2e-2,1:1},{0:5e-2,1:1},\
                                    {0:1e-1,1:1},{0:2e-1,1:1},{0:5e-1,1:1},{0:1e0,1:1}]}
    base_mdl_svm=SVC(shrinking=False,\
                        probability=False,random_state=0,max_iter=-1,\
                            tol=X_CV.shape[-1]*1e-3)
    
    clf_svm = GridSearchCV(base_mdl_svm, param_grid_svm,scoring='roc_auc',\
                       n_jobs=-1,cv=k_fold,refit=False)
    clf_svm.fit(X_CV, Y_CV)
    opt_params_svm=clf_svm.best_params_
    C_opt=opt_params_svm['class_weight'][0]
    kernel_opt=opt_params_svm['kernel']
    score_svm=clf_svm.best_score_
    
    t2=datetime.now()
    dt=t2-t1
    print('SVM CV finished after '+str(dt.total_seconds())+' sec')
    print('SVM: The optimal C+/C- ratio is '+str(1/C_opt))
    
    # optimise Logistic Regression –Dechow et al (2011)
    print('Grid search hyperparameter optimisation started for LR')
    t1=datetime.now()
    
    param_grid_lr={'class_weight':[{0:2e-3,1:1},{0:5e-3,1:1},{0:1e-2,1:1},{0:2e-2,1:1},{0:5e-2,1:1},\
                                    {0:1e-1,1:1},{0:2e-1,1:1},{0:5e-1,1:1},{0:1e0,1:1}]}
    base_mdl_lr=LogisticRegression(random_state=None)
    
    clf_lr = GridSearchCV(base_mdl_lr, param_grid_lr,scoring='roc_auc',\
                       n_jobs=-1,cv=k_fold,refit=False)
    clf_lr.fit(X_CV, Y_CV)
    opt_params_lr=clf_lr.best_params_
    score_lr=clf_lr.best_score_
    C_opt_lr=opt_params_lr['class_weight'][0]
    t2=datetime.now()
    dt=t2-t1
    print('LR CV finished after '+str(dt.total_seconds())+' sec')
    print('LR: The optimal C+/C- ratio is '+str(1/C_opt_lr))
    
    # optimise SGD
    print('Grid search hyperparameter optimisation started for SGD')
    t1=datetime.now()
    
    param_grid_sgd={'penalty':['l1','l2'],'loss':['log','modified_huber'],\
                    'class_weight':[{0:2e-3,1:1},{0:5e-3,1:1},{0:1e-2,1:1},\
                                    {0:2e-2,1:1},{0:5e-2,1:1},{0:1e-1,1:1},\
                                        {0:2e-1,1:1},{0:5e-1,1:1},{0:1e0,1:1}]}
    base_mdl_sgd=SGDClassifier(random_state=0,validation_fraction=.2,shuffle=False)
    
    clf_sgd = GridSearchCV(base_mdl_sgd, param_grid_sgd,scoring='roc_auc',\
                       n_jobs=-1,cv=k_fold,refit=False)
    clf_sgd.fit(X_CV, Y_CV)
    opt_params_sgd=clf_sgd.best_params_
    score_sgd=clf_sgd.best_score_
    t2=datetime.now()
    dt=t2-t1
    print('SGD CV finished after '+str(dt.total_seconds())+' sec')
    print('SGD: The optimal C+/C- ratio is '+str(1/opt_params_sgd['class_weight'][0])+\
          ', loss function is '+opt_params_sgd['loss']+', and penalty '+\
              opt_params_sgd['penalty'])
    
    # optimise AdaBoost
    print('Grid search hyperparameter optimisation started for AdaBoost')
    t1=datetime.now()
    
    best_perf_ada=0
    class_weight_rng=[{0:2e-3,1:1},{0:5e-3,1:1},{0:1e-2,1:1},{0:2e-2,1:1},\
                      {0:5e-2,1:1},{0:1e-1,1:1},{0:2e-1,1:1},{0:5e-1,1:1},{0:1e0,1:1}]
    for class_ratio in class_weight_rng:
        param_grid_ada={'n_estimators':[10,20,50,100,200,500],\
                        'learning_rate':[.1,5e-1,9e-1,1]}
            
        base_lr=LogisticRegression(random_state=0,class_weight=class_ratio)
        base_mdl_ada=AdaBoostClassifier(base_estimator=base_lr,random_state=0)
        
        clf_ada = GridSearchCV(base_mdl_ada, param_grid_ada,scoring='roc_auc',\
                           n_jobs=-1,cv=k_fold,refit=False)
        clf_ada.fit(X_CV, Y_CV)
        score_ada=clf_ada.best_score_
        if score_ada>=best_perf_ada:
            best_perf_ada=score_ada
            opt_params_ada=clf_ada.best_params_
            opt_class_weight_ada_lr=class_ratio
        
    
    t2=datetime.now()
    dt=t2-t1
    print('ADA CV finished after '+str(dt.total_seconds())+' sec')
    
    print('ADA: The optimal number of estimators is '+\
          str(opt_params_ada['n_estimators'])+', and learning rate '+\
              str(opt_params_ada['learning_rate']))
    imbalance_fact=opt_class_weight_ada_lr[1]/opt_class_weight_ada_lr[0]
    print('ADA: The optimal C+/C- is '+str(imbalance_fact))
    
    # optimise MLP classifier
    print('Grid search hyperparameter optimisation started for MLP')
    t1=datetime.now()
    param_grid={'hidden_layer_sizes':[1,2,5,10],'solver':['sgd','adam'],\
                    'activation':['identity','logistic']}
    base_mdl_mlp=MLPClassifier(random_state=0,validation_fraction=.2)
    
    clf_mlp = GridSearchCV(base_mdl_mlp, param_grid,scoring='roc_auc',\
                       n_jobs=-1,cv=k_fold,refit=False)
    clf_mlp.fit(X_CV, Y_CV)
    opt_params=clf_mlp.best_params_
    score_mlp=clf_mlp.best_score_
    t2=datetime.now()
    dt=t2-t1
    print('MLP CV finished after '+str(dt.total_seconds())+' sec')
    print('MLP: The optimal number of hidden layer is '+\
          str(opt_params['hidden_layer_sizes'])+', activation function '+\
                      opt_params['activation']+', and solver '+\
                          opt_params['solver'])
    
    print('Hyperparameter optimisation finished successfully.\nStarting the main analysis now...')
else:
    n_opt=100
    r_opt=1e-3
    score_rus=0.695492921141167
    
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
    opt_class_weight_ada_lr={0:1e0,1:1}
    score_ada=0.700229450411913
    
    opt_params={'activation': 'logistic', 'hidden_layer_sizes': 5, 'solver': 'adam'}
    score_mlp=0.706333862286029

range_oos=range(start_OOS_year,end_OOS_year+1,OOS_period)

roc_rus=np.zeros(len(range_oos))
specificity_rus=np.zeros(len(range_oos))
sensitivity_OOS_rus=np.zeros(len(range_oos))
precision_rus=np.zeros(len(range_oos))
sensitivity_OOS_rus1=np.zeros(len(range_oos))
specificity_OOS_rus1=np.zeros(len(range_oos))
precision_rus1=np.zeros(len(range_oos))
ndcg_rus1=np.zeros(len(range_oos))
ecm_rus1=np.zeros(len(range_oos))


roc_svm=np.zeros(len(range_oos))
specificity_svm=np.zeros(len(range_oos))
sensitivity_OOS_svm=np.zeros(len(range_oos))
precision_svm=np.zeros(len(range_oos))
sensitivity_OOS_svm1=np.zeros(len(range_oos))
specificity_OOS_svm1=np.zeros(len(range_oos))
precision_svm1=np.zeros(len(range_oos))
ndcg_svm1=np.zeros(len(range_oos))
ecm_svm1=np.zeros(len(range_oos))

roc_lr=np.zeros(len(range_oos))
specificity_lr=np.zeros(len(range_oos))
sensitivity_OOS_lr=np.zeros(len(range_oos))
precision_lr=np.zeros(len(range_oos))
sensitivity_OOS_lr1=np.zeros(len(range_oos))
specificity_OOS_lr1=np.zeros(len(range_oos))
precision_lr1=np.zeros(len(range_oos))
ndcg_lr1=np.zeros(len(range_oos))
ecm_lr1=np.zeros(len(range_oos))

roc_sgd=np.zeros(len(range_oos))
specificity_sgd=np.zeros(len(range_oos))
sensitivity_OOS_sgd=np.zeros(len(range_oos))
precision_sgd=np.zeros(len(range_oos))
sensitivity_OOS_sgd1=np.zeros(len(range_oos))
specificity_OOS_sgd1=np.zeros(len(range_oos))
precision_sgd1=np.zeros(len(range_oos))
ndcg_sgd1=np.zeros(len(range_oos))
ecm_sgd1=np.zeros(len(range_oos))

roc_ada=np.zeros(len(range_oos))
specificity_ada=np.zeros(len(range_oos))
sensitivity_OOS_ada=np.zeros(len(range_oos))
precision_ada=np.zeros(len(range_oos))
sensitivity_OOS_ada1=np.zeros(len(range_oos))
specificity_OOS_ada1=np.zeros(len(range_oos))
precision_ada1=np.zeros(len(range_oos))
ndcg_ada1=np.zeros(len(range_oos))
ecm_ada1=np.zeros(len(range_oos))


roc_mlp=np.zeros(len(range_oos))
specificity_mlp=np.zeros(len(range_oos))
sensitivity_OOS_mlp=np.zeros(len(range_oos))
precision_mlp=np.zeros(len(range_oos))
sensitivity_OOS_mlp1=np.zeros(len(range_oos))
specificity_OOS_mlp1=np.zeros(len(range_oos))
precision_mlp1=np.zeros(len(range_oos))
ndcg_mlp1=np.zeros(len(range_oos))
ecm_mlp1=np.zeros(len(range_oos))


roc_fused=np.zeros(len(range_oos))
specificity_fused=np.zeros(len(range_oos))
sensitivity_OOS_fused=np.zeros(len(range_oos))
precision_fused=np.zeros(len(range_oos))
sensitivity_OOS_fused1=np.zeros(len(range_oos))
specificity_OOS_fused1=np.zeros(len(range_oos))
precision_fused1=np.zeros(len(range_oos))
ndcg_fused1=np.zeros(len(range_oos))
ecm_fused1=np.zeros(len(range_oos))


m=0
for yr in range_oos:
    t1=datetime.now()
    if case_window=='expanding':
        year_start_IS=1991
    else:
        year_start_IS=yr-IS_period
    
    tbl_year_IS=reduced_tbl.loc[np.logical_and(reduced_tbl.fyear<yr-OOS_gap,\
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
    
    
    # RUSBoost with 11 ratios
    
    base_tree=DecisionTreeClassifier(min_samples_leaf=5)
    bao_RUSboost=RUSBoostClassifier(base_estimator=base_tree,n_estimators=n_opt,\
                     learning_rate=r_opt,sampling_strategy=1,random_state=0)
    clf_rusboost=bao_RUSboost.fit(X, Y)
    probs_oos_fraud_rus=clf_rusboost.predict_proba(X_OOS)[:,-1]
    roc_rus[m]=roc_auc_score(Y_OOS,probs_oos_fraud_rus)
    
    labels_rus=clf_rusboost.predict(X_OOS)
    
    roc_rus[m]=roc_auc_score(Y_OOS,probs_oos_fraud_rus)
    specificity_rus[m]=np.sum(np.logical_and(labels_rus==0,Y_OOS==0))/\
        np.sum(Y_OOS==0)
    if np.sum(labels_rus)>0:
        sensitivity_OOS_rus[m]=np.sum(np.logical_and(labels_rus==1, \
                                                 Y_OOS==1))/np.sum(Y_OOS)
        precision_rus[m]=np.sum(np.logical_and(labels_rus==1,Y_OOS==1))/np.sum(labels_rus)
    
    
    cutoff_OOS_rus=np.percentile(probs_oos_fraud_rus,99)
    sensitivity_OOS_rus1[m]=np.sum(np.logical_and(probs_oos_fraud_rus>=cutoff_OOS_rus, \
                                                  Y_OOS==1))/np.sum(Y_OOS)
    specificity_OOS_rus1[m]=np.sum(np.logical_and(probs_oos_fraud_rus<cutoff_OOS_rus, \
                                                  Y_OOS==0))/np.sum(Y_OOS==0)
    precision_rus1[m]=np.sum(np.logical_and(probs_oos_fraud_rus>=cutoff_OOS_rus, \
                                                 Y_OOS==1))/np.sum(probs_oos_fraud_rus>=cutoff_OOS_rus)
    ndcg_rus1[m]=ndcg_k(Y_OOS,probs_oos_fraud_rus,99)
    
    FN_rus1=np.sum(np.logical_and(probs_oos_fraud_rus<cutoff_OOS_rus, \
                                                  Y_OOS==1))
    FP_rus1=np.sum(np.logical_and(probs_oos_fraud_rus>=cutoff_OOS_rus, \
                                                  Y_OOS==0))
        
    ecm_rus1[m]=C_FN*P_f*FN_rus1/n_P+C_FP*P_nf*FP_rus1/n_N
    
    
    # Support Vector Machines
    
    clf_svm=SVC(class_weight={0:C_opt,1:1},kernel=kernel_opt,shrinking=False,\
                    probability=False,random_state=0,max_iter=-1,\
                        tol=X.shape[-1]*1e-3)
        
    clf_svm=clf_svm.fit(X,Y)
    
    pred_test_svm=clf_svm.decision_function(X_OOS)
    probs_oos_fraud_svm=np.exp(pred_test_svm)/(1+np.exp(pred_test_svm))
    

    
    labels_svm=clf_svm.predict(X_OOS)
    
    roc_svm[m]=roc_auc_score(Y_OOS,probs_oos_fraud_svm)
    specificity_svm[m]=np.sum(np.logical_and(labels_svm==0,Y_OOS==0))/\
        np.sum(Y_OOS==0)
    if np.sum(labels_svm)>0:
        sensitivity_OOS_svm[m]=np.sum(np.logical_and(labels_svm==1, \
                                                 Y_OOS==1))/np.sum(Y_OOS)
        precision_svm[m]=np.sum(np.logical_and(labels_svm==1,Y_OOS==1))/np.sum(labels_svm)
    
    
    cutoff_OOS_svm=np.percentile(probs_oos_fraud_svm,99)
    sensitivity_OOS_svm1[m]=np.sum(np.logical_and(probs_oos_fraud_svm>=cutoff_OOS_svm, \
                                                  Y_OOS==1))/np.sum(Y_OOS)
    specificity_OOS_svm1[m]=np.sum(np.logical_and(probs_oos_fraud_svm<cutoff_OOS_svm, \
                                                  Y_OOS==0))/np.sum(Y_OOS==0)
    precision_svm1[m]=np.sum(np.logical_and(probs_oos_fraud_svm>=cutoff_OOS_svm, \
                                                 Y_OOS==1))/np.sum(probs_oos_fraud_svm>=cutoff_OOS_svm)
    ndcg_svm1[m]=ndcg_k(Y_OOS,probs_oos_fraud_svm,99)
    
    FN_svm1=np.sum(np.logical_and(probs_oos_fraud_svm<cutoff_OOS_svm, \
                                                  Y_OOS==1))
    FP_svm1=np.sum(np.logical_and(probs_oos_fraud_svm>=cutoff_OOS_svm, \
                                                  Y_OOS==0))
        
    ecm_svm1[m]=C_FN*P_f*FN_svm1/n_P+C_FP*P_nf*FP_svm1/n_N
        
    
    # Logistic Regression – Dechow et al (2011)
    
    clf_lr = LogisticRegression(class_weight=opt_params_lr['class_weight'],\
                             random_state=None).fit(X,Y)

    probs_oos_lr=clf_lr.predict_proba(X_OOS)
    probs_oos_fraud_lr=probs_oos_lr[:,-1]

    labels_lr=clf_lr.predict(X_OOS)

    roc_lr[m]=roc_auc_score(Y_OOS,probs_oos_fraud_lr)
    
    specificity_lr[m]=np.sum(np.logical_and(labels_lr==0,Y_OOS==0))/\
        np.sum(Y_OOS==0)
    if np.sum(labels_lr)>0:
        sensitivity_OOS_lr[m]=np.sum(np.logical_and(labels_lr==1, \
                                                 Y_OOS==1))/np.sum(Y_OOS)
        precision_lr[m]=np.sum(np.logical_and(labels_lr==1,Y_OOS==1))/np.sum(labels_lr)
    
    cutoff_OOS_lr=np.percentile(probs_oos_fraud_lr,99)
    sensitivity_OOS_lr1[m]=np.sum(np.logical_and(probs_oos_fraud_lr>=cutoff_OOS_lr, \
                                                 Y_OOS==1))/np.sum(Y_OOS)
    specificity_OOS_lr1[m]=np.sum(np.logical_and(probs_oos_fraud_lr<cutoff_OOS_lr, \
                                                  Y_OOS==0))/np.sum(Y_OOS==0)
    precision_lr1[m]=np.sum(np.logical_and(probs_oos_fraud_lr>=cutoff_OOS_lr, \
                                                 Y_OOS==1))/np.sum(probs_oos_fraud_lr>=cutoff_OOS_lr)
    ndcg_lr1[m]=ndcg_k(Y_OOS,probs_oos_fraud_lr,99)
    
    FN_lr1=np.sum(np.logical_and(probs_oos_fraud_lr<cutoff_OOS_lr, \
                                                  Y_OOS==1))
    FP_lr1=np.sum(np.logical_and(probs_oos_fraud_lr>=cutoff_OOS_lr, \
                                                  Y_OOS==0))
        
    ecm_lr1[m]=C_FN*P_f*FN_lr1/n_P+C_FP*P_nf*FP_lr1/n_N
        
    
    # Stochastic Gradient Decent 

    clf_sgd=SGDClassifier(class_weight=opt_params_sgd['class_weight'],\
                          loss=opt_params_sgd['loss'], random_state=0,\
                           penalty=opt_params_sgd['penalty'],validation_fraction=.2,shuffle=False)
    clf_sgd=clf_sgd.fit(X,Y)
    probs_oos_fraud_sgd=clf_sgd.predict_proba(X_OOS)[:,-1]
    
    
    labels_sgd=clf_sgd.predict(X_OOS)
    
    roc_sgd[m]=roc_auc_score(Y_OOS,probs_oos_fraud_sgd)
    specificity_sgd[m]=np.sum(np.logical_and(labels_sgd==0,Y_OOS==0))/\
        np.sum(Y_OOS==0)
    if np.sum(labels_sgd)>0:
        sensitivity_OOS_sgd[m]=np.sum(np.logical_and(labels_sgd==1, \
                                                 Y_OOS==1))/np.sum(Y_OOS)
        precision_sgd[m]=np.sum(np.logical_and(labels_sgd==1,Y_OOS==1))/np.sum(labels_sgd)
    
    
    cutoff_OOS_sgd=np.percentile(probs_oos_fraud_sgd,99)
    sensitivity_OOS_sgd1[m]=np.sum(np.logical_and(probs_oos_fraud_sgd>=cutoff_OOS_sgd, \
                                                 Y_OOS==1))/np.sum(Y_OOS)
    specificity_OOS_sgd1[m]=np.sum(np.logical_and(probs_oos_fraud_sgd<cutoff_OOS_sgd, \
                                                  Y_OOS==0))/np.sum(Y_OOS==0)
    precision_sgd1[m]=np.sum(np.logical_and(probs_oos_fraud_sgd>=cutoff_OOS_sgd, \
                                                 Y_OOS==1))/np.sum(probs_oos_fraud_sgd>=cutoff_OOS_sgd)
    ndcg_sgd1[m]=ndcg_k(Y_OOS,probs_oos_fraud_sgd,99)
    
    FN_sgd1=np.sum(np.logical_and(probs_oos_fraud_sgd<cutoff_OOS_sgd, \
                                                  Y_OOS==1))
    FP_sgd1=np.sum(np.logical_and(probs_oos_fraud_sgd>=cutoff_OOS_sgd, \
                                                  Y_OOS==0))
        
    ecm_sgd1[m]=C_FN*P_f*FN_sgd1/n_P+C_FP*P_nf*FP_sgd1/n_N
    
    
    
    
    # LogitBoost
    base_lr=LogisticRegression(random_state=0)
    
    clf_ada=AdaBoostClassifier(n_estimators=opt_params_ada['n_estimators'],\
                               learning_rate=opt_params_ada['learning_rate'],\
                                   base_estimator=base_lr,random_state=0)
    clf_ada=clf_ada.fit(X,Y)
    probs_oos_fraud_ada=clf_ada.predict_proba(X_OOS)[:,-1]
    
    
    labels_ada=clf_ada.predict(X_OOS)
    
    roc_ada[m]=roc_auc_score(Y_OOS,probs_oos_fraud_ada)
    specificity_ada[m]=np.sum(np.logical_and(labels_ada==0,Y_OOS==0))/\
        np.sum(Y_OOS==0)
    if np.sum(labels_ada)>0:
        sensitivity_OOS_ada[m]=np.sum(np.logical_and(labels_ada==1, \
                                                 Y_OOS==1))/np.sum(Y_OOS)
        precision_ada[m]=np.sum(np.logical_and(labels_ada==1,Y_OOS==1))/np.sum(labels_ada)
    
    cutoff_OOS_ada=np.percentile(probs_oos_fraud_ada,99)
    sensitivity_OOS_ada1[m]=np.sum(np.logical_and(probs_oos_fraud_ada>=cutoff_OOS_ada, \
                                                 Y_OOS==1))/np.sum(Y_OOS)
    specificity_OOS_ada1[m]=np.sum(np.logical_and(probs_oos_fraud_ada<cutoff_OOS_ada, \
                                                  Y_OOS==0))/np.sum(Y_OOS==0)
    precision_ada1[m]=np.sum(np.logical_and(probs_oos_fraud_ada>=cutoff_OOS_ada, \
                                                 Y_OOS==1))/np.sum(probs_oos_fraud_ada>=cutoff_OOS_ada)
    ndcg_ada1[m]=ndcg_k(Y_OOS,probs_oos_fraud_ada,99)
    
    FN_ada1=np.sum(np.logical_and(probs_oos_fraud_ada<cutoff_OOS_ada, \
                                                  Y_OOS==1))
    FP_ada1=np.sum(np.logical_and(probs_oos_fraud_ada>=cutoff_OOS_ada, \
                                                  Y_OOS==0))
        
    ecm_ada1[m]=C_FN*P_f*FN_ada1/n_P+C_FP*P_nf*FP_ada1/n_N
        
    
    # Multi Layer Perceptron
    clf_mlp=MLPClassifier(hidden_layer_sizes=opt_params['hidden_layer_sizes'], \
                          activation=opt_params['activation'],solver=opt_params['solver'],\
                                       random_state=0,validation_fraction=.1)
    clf_mlp=clf_mlp.fit(X,Y)
    probs_oos_fraud_mlp=clf_mlp.predict_proba(X_OOS)[:,-1]
    
    labels_mlp=clf_mlp.predict(X_OOS)
    
    roc_mlp[m]=roc_auc_score(Y_OOS,probs_oos_fraud_mlp)
    specificity_mlp[m]=np.sum(np.logical_and(labels_mlp==0,Y_OOS==0))/\
        np.sum(Y_OOS==0)
    
    if np.sum(labels_mlp)>0:
        sensitivity_OOS_mlp[m]=np.sum(np.logical_and(labels_mlp==1, \
                                                 Y_OOS==1))/np.sum(Y_OOS)
        precision_mlp[m]=np.sum(np.logical_and(labels_mlp==1,Y_OOS==1))/np.sum(labels_mlp)
    
    cutoff_OOS_mlp=np.percentile(probs_oos_fraud_mlp,99)
    sensitivity_OOS_mlp1[m]=np.sum(np.logical_and(probs_oos_fraud_mlp>=cutoff_OOS_mlp, \
                                                 Y_OOS==1))/np.sum(Y_OOS)
    specificity_OOS_mlp1[m]=np.sum(np.logical_and(probs_oos_fraud_mlp<cutoff_OOS_mlp, \
                                                  Y_OOS==0))/np.sum(Y_OOS==0)
    precision_mlp1[m]=np.sum(np.logical_and(probs_oos_fraud_mlp>=cutoff_OOS_mlp, \
                                                 Y_OOS==1))/np.sum(probs_oos_fraud_mlp>=cutoff_OOS_mlp)
    ndcg_mlp1[m]=ndcg_k(Y_OOS,probs_oos_fraud_mlp,99)
    
    FN_mlp1=np.sum(np.logical_and(probs_oos_fraud_mlp<cutoff_OOS_mlp, \
                                                  Y_OOS==1))
    FP_mlp1=np.sum(np.logical_and(probs_oos_fraud_mlp>=cutoff_OOS_mlp, \
                                                  Y_OOS==0))
        
    ecm_mlp1[m]=C_FN*P_f*FN_mlp1/n_P+C_FP*P_nf*FP_mlp1/n_N
        
    
    
    # Fused approach
    
    weight_ser=np.array([score_svm,score_lr,score_sgd,score_ada,score_mlp])
    weight_ser=weight_ser/np.sum(weight_ser)
    
    clf_fused=np.dot(np.array([probs_oos_fraud_svm,\
                      probs_oos_fraud_lr,\
                          probs_oos_fraud_sgd,probs_oos_fraud_ada,\
                              probs_oos_fraud_mlp]).T,weight_ser)
    
    probs_oos_fraud_fused=clf_fused
    
    labels_fused=(clf_fused>=np.percentile(clf_fused,99)).astype(int)
    
    roc_fused[m]=roc_auc_score(Y_OOS,probs_oos_fraud_fused)
    specificity_fused[m]=np.sum(np.logical_and(labels_fused==0,Y_OOS==0))/\
        np.sum(Y_OOS==0)
    
    if np.sum(labels_fused)>0:
        sensitivity_OOS_fused[m]=np.sum(np.logical_and(labels_fused==1, \
                                                 Y_OOS==1))/np.sum(Y_OOS)
        precision_fused[m]=np.sum(np.logical_and(labels_fused==1,Y_OOS==1))/np.sum(labels_fused)
    
    cutoff_OOS_fused=np.percentile(probs_oos_fraud_fused,99)
    sensitivity_OOS_fused1[m]=np.sum(np.logical_and(probs_oos_fraud_fused>=cutoff_OOS_fused, \
                                                 Y_OOS==1))/np.sum(Y_OOS)
    specificity_OOS_fused1[m]=np.sum(np.logical_and(probs_oos_fraud_fused<cutoff_OOS_fused, \
                                                  Y_OOS==0))/np.sum(Y_OOS==0)
    precision_fused1[m]=np.sum(np.logical_and(probs_oos_fraud_fused>=cutoff_OOS_fused, \
                                                 Y_OOS==1))/np.sum(probs_oos_fraud_fused>=cutoff_OOS_fused)
    ndcg_fused1[m]=ndcg_k(Y_OOS,probs_oos_fraud_fused,99)
    
    FN_fused1=np.sum(np.logical_and(probs_oos_fraud_fused<cutoff_OOS_fused, \
                                                  Y_OOS==1))
    FP_fused1=np.sum(np.logical_and(probs_oos_fraud_fused>=cutoff_OOS_fused, \
                                                  Y_OOS==0))
        
    ecm_fused1[m]=C_FN*P_f*FN_fused1/n_P+C_FP*P_nf*FP_fused1/n_N
    

    
    t2=datetime.now() 
    dt=t2-t1
    print('analysis finished for OOS period '+str(yr)+' after '+str(dt.total_seconds())+' sec')
    m+=1

print('average top percentile sensitivity for the period '+str(start_OOS_year)+' to '+\
      str(end_OOS_year)+' is '+ str(round(np.mean(sensitivity_OOS_rus1)*100,2))+\
              '% for RUSBoost vs '+ str(round(np.mean(sensitivity_OOS_svm1)*100,2))+\
                  '% for SVM vs '+ str(round(np.mean(sensitivity_OOS_lr1)*100,2))+\
              '% for Dechow-LR vs '+ str(round(np.mean(sensitivity_OOS_sgd1)*100,2))+\
                      '% for SGD vs '+ str(round(np.mean(sensitivity_OOS_ada1)*100,2))+\
                          '% for ADA vs '+ str(round(np.mean(sensitivity_OOS_mlp1)*100,2))+\
                              '% for MLP vs '+ str(round(np.mean(sensitivity_OOS_fused1)*100,2))+\
                                  '% for FUSED')


# create performance table now
perf_tbl_general=pd.DataFrame()
perf_tbl_general['models']=['RUSBoost','SVM','LR','SGD','ADA','MLP','FUSED']
perf_tbl_general['Roc']=[np.mean(roc_rus),np.mean(roc_svm),np.mean(roc_lr),\
                         np.mean(roc_sgd),np.mean(roc_ada),\
                             np.mean(roc_mlp),np.mean(roc_fused)]

                                            
perf_tbl_general['Sensitivity @ 1 Prc']=[np.mean(sensitivity_OOS_rus1),\
                                         np.mean(sensitivity_OOS_svm1),\
                                 np.mean(sensitivity_OOS_lr1),\
                                     np.mean(sensitivity_OOS_sgd1),np.mean(sensitivity_OOS_ada1),\
                                         np.mean(sensitivity_OOS_mlp1),np.mean(sensitivity_OOS_fused1)]

perf_tbl_general['Specificity @ 1 Prc']=[np.mean(specificity_OOS_rus1),\
                                         np.mean(specificity_OOS_svm1),\
                                 np.mean(specificity_OOS_lr1),\
                                     np.mean(specificity_OOS_sgd1),np.mean(specificity_OOS_ada1),\
                                         np.mean(specificity_OOS_mlp1),np.mean(specificity_OOS_fused1)]

perf_tbl_general['Precision @ 1 Prc']=[np.mean(precision_rus1),\
                                       np.mean(precision_svm1),\
                                 np.mean(precision_lr1),\
                                     np.mean(precision_sgd1),np.mean(precision_ada1),\
                                         np.mean(precision_mlp1),np.mean(precision_fused1)]

perf_tbl_general['F1 Score @ 1 Prc']=2*(perf_tbl_general['Precision @ 1 Prc']*\
                                      perf_tbl_general['Sensitivity @ 1 Prc'])/\
                                        ((perf_tbl_general['Precision @ 1 Prc']+\
                                          perf_tbl_general['Sensitivity @ 1 Prc']))
perf_tbl_general['NDCG @ 1 Prc']=[np.mean(ndcg_rus1),np.mean(ndcg_svm1),\
                                 np.mean(ndcg_lr1),\
                                     np.mean(ndcg_sgd1),np.mean(ndcg_ada1),\
                                         np.mean(ndcg_mlp1),np.mean(ndcg_fused1)]

perf_tbl_general['ECM @ 1 Prc']=[np.mean(ecm_rus1),np.mean(ecm_svm1),\
                                 np.mean(ecm_lr1),\
                                     np.mean(ecm_sgd1),np.mean(ecm_ada1),\
                                         np.mean(ecm_mlp1),np.mean(ecm_fused1)]



# perf_tbl_general['F.5 Score @ 1 Prc']=(1+np.power(.5,2))*(perf_tbl_general['Precision @ 1 Prc']*\
#                                       perf_tbl_general['Sensitivity @ 1 Prc'])/\
#                                         ((perf_tbl_general['Precision @ 1 Prc']*np.power(.5,2)+\
#                                           perf_tbl_general['Sensitivity @ 1 Prc']))
    
# perf_tbl_general['F2 Score @ 1 Prc']=(1+np.power(2,2))*(perf_tbl_general['Precision @ 1 Prc']*\
#                                       perf_tbl_general['Sensitivity @ 1 Prc'])/\
#                                         ((perf_tbl_general['Precision @ 1 Prc']*np.power(2,2)+\
#                                           perf_tbl_general['Sensitivity @ 1 Prc']))
    

if case_window=='expanding':
    lbl_perf_tbl='perf_tbl_'+str(start_OOS_year)+'_'+str(end_OOS_year)+\
        '_'+case_window+',OOS='+str(OOS_period)+','+\
        str(k_fold)+'fold'+',serial='+str(adjust_serial)+\
        ',gap='+str(OOS_gap)+'_11ratios.csv'
else:
    lbl_perf_tbl='perf_tbl_'+str(start_OOS_year)+'_'+str(end_OOS_year)+\
        '_IS='+str(IS_period)+',OOS='+str(OOS_period)+','+\
        str(k_fold)+'fold'+',serial='+str(adjust_serial)+\
        ',gap='+str(OOS_gap)+'_11ratios.csv'

perf_tbl_general.to_csv(lbl_perf_tbl,index=False)
print(perf_tbl_general)
t_last=datetime.now()
dt_total=t_last-t0
print('total run time is '+str(dt_total.total_seconds())+' sec')


# extract performance for 2003-2008 directly from 2001-2010

perf_tbl_general=pd.DataFrame()
perf_tbl_general['models']=['RUSBoost','SVM','LR','SGD','ADA','MLP','FUSED']
perf_tbl_general['Roc']=[np.mean(roc_rus[2:8]),np.mean(roc_svm[2:8]),np.mean(roc_lr[2:8]),\
                         np.mean(roc_sgd[2:8]),\
                             np.mean(roc_ada[2:8]),np.mean(roc_mlp[2:8]),np.mean(roc_fused[2:8])]

                                            
perf_tbl_general['Sensitivity @ 1 Prc']=[np.mean(sensitivity_OOS_rus1[2:8]),\
                                         np.mean(sensitivity_OOS_svm1[2:8]),\
                                 np.mean(sensitivity_OOS_lr1[2:8]),\
                                     np.mean(sensitivity_OOS_sgd1[2:8]),\
                                         np.mean(sensitivity_OOS_ada1[2:8]),\
                                         np.mean(sensitivity_OOS_mlp1[2:8]),\
                                             np.mean(sensitivity_OOS_fused1[2:8])]

perf_tbl_general['Specificity @ 1 Prc']=[np.mean(specificity_OOS_rus1[2:8]),\
                                         np.mean(specificity_OOS_svm1[2:8]),\
                                 np.mean(specificity_OOS_lr1[2:8]),\
                                     np.mean(specificity_OOS_sgd1[2:8]),\
                                         np.mean(specificity_OOS_ada1[2:8]),\
                                         np.mean(specificity_OOS_mlp1[2:8]),\
                                             np.mean(specificity_OOS_fused1[2:8])]

perf_tbl_general['Precision @ 1 Prc']=[np.mean(precision_rus1[2:8]),\
                                       np.mean(precision_svm1[2:8]),\
                                 np.mean(precision_lr1[2:8]),\
                                     np.mean(precision_sgd1[2:8]),\
                                         np.mean(precision_ada1[2:8]),\
                                         np.mean(precision_mlp1[2:8]),\
                                             np.mean(precision_fused1[2:8])]

perf_tbl_general['F1 Score @ 1 Prc']=2*(perf_tbl_general['Precision @ 1 Prc']*\
                                      perf_tbl_general['Sensitivity @ 1 Prc'])/\
                                        ((perf_tbl_general['Precision @ 1 Prc']+\
                                          perf_tbl_general['Sensitivity @ 1 Prc']))


perf_tbl_general['NDCG @ 1 Prc']=[np.mean(ndcg_rus1[2:8]),\
                                  np.mean(ndcg_svm1[2:8]),\
                                 np.mean(ndcg_svm1[2:8]),\
                                     np.mean(ndcg_sgd1[2:8]),np.mean(ndcg_ada1[2:8]),\
                                         np.mean(ndcg_mlp1[2:8]),\
                                             np.mean(ndcg_fused1[2:8])]
perf_tbl_general['ECM @ 1 Prc']=[np.mean(ecm_rus1[2:8]),\
                                 np.mean(ecm_svm1[2:8]),\
                                 np.mean(ecm_lr1[2:8]),\
                                     np.mean(ecm_sgd1[2:8]),np.mean(ecm_ada1[2:8]),\
                                         np.mean(ecm_mlp1[2:8]),np.mean(ecm_fused1[2:8])]

if case_window=='expanding':
    lbl_perf_tbl='perf_tbl_'+str(2003)+'_'+str(2008)+\
        '_'+case_window+',OOS='+str(OOS_period)+','+\
        str(k_fold)+'fold'+',serial='+str(adjust_serial)+\
        ',gap='+str(OOS_gap)+'_11ratios.csv'
else:
    lbl_perf_tbl='perf_tbl_'+str(2003)+'_'+str(2008)+\
        '_IS='+str(IS_period)+',OOS='+str(OOS_period)+','+\
        str(k_fold)+'fold'+',serial='+str(adjust_serial)+\
            ',gap='+str(OOS_gap)+'_11ratios.csv'
perf_tbl_general.to_csv(lbl_perf_tbl,index=False)
