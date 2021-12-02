#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 12:24:27 2021

This code replicates the SVM with Financial Kernel as in Cecchini et al (2010)
https://doi.org/10.1287/mnsc.1100.1174
This code uses 23 raw financial variables and generates 1518 features out of 
the initial variable set. 
Inputs: 
    – A csv file under name "FraudDB2020.csv" with information regarding 
    US companies and AAER status. The column "AAER_DUMMY" is the identifier of the
    firms with a fraud case. 
    Expected columns are:     
    0    fyear, 1    gvkey, 2    sich, 3    indfmt,4    cik, 5    datadate,6    fic,7    AAER_OVER,
    8    AAER_DUMMY, 9    act, 10    ap, 11    at, 12    ceq, 13    che, 14    cogs, 15    csho,
    16    dlc, 17    dltis, 18    dltt, 19    dp, 20    ib, 21    invt, 22    ivao, 23    ivst,
    24    lct, 25    lt, 26    ni, 27    ppegt, 28    pstk, 29    re, 30    rect, 31    sale,
    32    sstk, 33    txc, 34    txp, 35    xint, 36    prcc_f, 37    dwc, 38    rsst, 39    dreceivables,
    40    dinventory, 41    soft_assets, 42    chcs, 43    chcashmargin, 44    droa,
    45    dfcf, 46    issuance, 47    bm, 48    depindex, 49    ebitat, 50    reat
    
Parameters:
    – cross_val:True/False (default) choice between using C+/C-=100
    or perform a cross-validation.
    – record_matrix: True/False (default) choice between recording the feature matrix
    and the lagged data table into a pickle file to allow further runs more quickly.
    If True a pickle file of size 900MB is going to be stored on your disk.
    – adjust_serial: True (default)/False choice whether to discard the firms with previous 
    fraud records in the OOS analysis. 
    – OOS_period: number of years in each out-of-sample prediction practice.
    – IS_period: number of years for training.

Outputs: 
Main results are stored in the table variable "perf_tbl_general" written into
2 csv files: time period 2001-2010 and 2003-2008

Steps:
    1. Generate lagged data. For each reported financial figure for each unique
    company the script finds the last observation and records the last amount. 
    Accordingly for the set of 23 inputs we have 46 inputs in "tbl_ratio_fk".
    2. Create the feature space based on Financial Kernel as in Cecchini et al (2010).
    The mapped feature space is stored in variable "mapped_X" with 1518 attributes.
    3. 5/10-fold Cross-validation to find optimal C+/C- ratio (optional)
    4. Estimating the performance for each OOS period.

Warnings: 
    – Running this code can take several hours. The generation of lagged data 
    takes ~40 mins, mapped matrix of inputs (N,1518) ~320 mins, CV ~ 40 mins,
    and main analysis ~ 120 mins. These figures are estimates based on a MacBook Pro 2017.
    – To make sure the computations are stored, you can set variable "record_matrix" to True.
    – If "record_matrix==True" the code stores a pickle file of ~ 900MB on your disk.
    – You can choose to turn cross-validation on/off by setting "cross_val" to True/False
    – If "cross_val==True" a five-fold cross-validation is performed. 

@author: Arman Hassanniakalager GitHub: https://github.com/hkalager
Common disclaimers apply. Subject to change at all time.

Last review: 22/10/2021
"""


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from sklearn.metrics import roc_auc_score
from extra_codes import ndcg_k
import pickle
import os

# start the clock!
t0=datetime.now()
# number of 
OOS_period=1
IS_period=10
k_fold=10
start_OOS_year=2001
end_OOS_year=2010
C_FN=30 #relative to C_FP
C_FP=1
cross_val=False
record_matrix=False
adjust_serial=True
case_window='expanding'
fraud_df=pd.read_csv('FraudDB2020.csv')
# First discard five variables that are in Bao et al (2020)
# but not in Cecchini et al (2010)
fraud_df.pop('act')
fraud_df.pop('ap')
fraud_df.pop('ppegt')
fraud_df.pop('dltis')
fraud_df.pop('sstk')

reduced_tbl_1=fraud_df.iloc[:,[0,1,3,7,8]]
reduced_tbl_2=fraud_df.iloc[:,9:-14]
reduced_tblset=[reduced_tbl_1,reduced_tbl_2]
reduced_tbl=pd.concat(reduced_tblset,axis=1)
reduced_tbl=reduced_tbl.reset_index(drop=True)

if os.path.isfile('features_fk.pkl')!=True:
    print('No pickle file found ...')
    print('processing data to extract last years financial figures')
    t1=datetime.now()
    tbl_ratio_fk=reduced_tbl
    new_cols=tbl_ratio_fk.columns[-23:]+'_last'
    addtbl=pd.DataFrame(columns=new_cols)
    progress_unit=len(tbl_ratio_fk)//20
    for m in range(0,len(tbl_ratio_fk)):
        if np.mod(m,progress_unit)==0:
            print(str(int(m//progress_unit*100/20))+'% lagged data processing completed')
        sel_gvkey=tbl_ratio_fk.gvkey[m]
        data_gvkey=tbl_ratio_fk[tbl_ratio_fk['gvkey']==sel_gvkey]
        current_idx=np.where(data_gvkey.index==m)[0][0]
        if current_idx>0:
            last_data=data_gvkey.iloc[current_idx-1,-23:]
            addtbl.loc[m,:]=last_data.values
        else:
            last_data=np.ones_like(data_gvkey.iloc[current_idx,-23:])*float('nan')
            addtbl.loc[m,:]=last_data
            
    
    dt=round((datetime.now()-t1).total_seconds()/60,3)
    print('processing data to extract last years financial figures')
    print('elapsed time '+str(dt)+' mins')
    tbl_ratio_fk=pd.concat([tbl_ratio_fk,addtbl],axis=1)
    tbl_ratio_fk=tbl_ratio_fk[np.logical_and(tbl_ratio_fk.fyear>=1990,tbl_ratio_fk.fyear<=2010)]
    init_size_tbl=len(tbl_ratio_fk)
    tbl_ratio_fk=tbl_ratio_fk[tbl_ratio_fk.at_last.isna()==False]
    tbl_ratio_fk=tbl_ratio_fk.reset_index(drop=True)
    drop_for_missing=init_size_tbl/len(tbl_ratio_fk)-1
    print(str(round(drop_for_missing*100,2))+'% of observations dropped due to '+\
          'missing data from last year')
    
    # Map the raw fundamentals into features >>> 3n(n-1) features for n attributes
    # this means 1518 features for 23 attributes originally
    t1=datetime.now()
    red_tbl_fk=tbl_ratio_fk.iloc[:,-46:]
    mapped_X=np.array([])
    progress_unit2=len(red_tbl_fk)//100
    for m in range(0,len(red_tbl_fk)):
        if np.mod(m,progress_unit2)==0:
            print(str(int(m//progress_unit2*100/100))+'% feature processing completed')
        features=np.array([])
        for i in range(0,23):
            for j in range(i+1,23):
                ui1=red_tbl_fk.iloc[m,i]
                ui1=(ui1==0).astype(int)*1e-4+ui1
                ui2=red_tbl_fk.iloc[m,23+i]
                ui2=(ui2==0).astype(int)*1e-4+ui2
                uj1=red_tbl_fk.iloc[m,j]
                uj1=(uj1==0).astype(int)*1e-4+uj1
                uj2=red_tbl_fk.iloc[m,23+j]
                uj2=(uj2==0).astype(int)*1e-4+uj2
                features_new=np.array([ui1/uj1,uj1/ui1,uj2/ui2,ui2/uj2,(ui1*uj2)/(uj1*ui2),\
                          (uj1*ui2)/(ui1*uj2)])
                features=np.append(features,features_new)
        if mapped_X.shape[-1]==0:
            mapped_X=np.append(mapped_X,features.T)
        else:
            mapped_X=np.vstack((mapped_X,features.T))
        
    dt=round((datetime.now()-t1).total_seconds()/60,3)
    print('feature processing completed ...')
    print('elapsed time '+str(dt)+' mins')
    if record_matrix==True:
        DB_Dict={'matrix':mapped_X,'lagged_Data':tbl_ratio_fk}
        fl_name='features_fk.pkl'
    # Write into a file
        pickle.dump(DB_Dict, open(fl_name,'w+b'))
else:
    print('pickle file available ...')
    dict_db=pickle.load(open('features_fk.pkl','r+b'))
    tbl_ratio_fk=dict_db['lagged_Data']
    mapped_X=dict_db['matrix']
    red_tbl_fk=tbl_ratio_fk.iloc[:,-46:]
    print('pickle file loaded successfully ...')
    
idx_CV=tbl_ratio_fk[np.logical_and(tbl_ratio_fk.fyear>=1991,tbl_ratio_fk.fyear<=2000)].index
Y_CV=tbl_ratio_fk.AAER_DUMMY[idx_CV]

X_CV=mapped_X[idx_CV,:]

idx_real=np.where(np.logical_and(np.isnan(X_CV).any(axis=1)==False,\
                                      np.isinf(X_CV).any(axis=1)==False))[0]
X_CV=X_CV[idx_real,:]

X_CV=(X_CV-np.mean(X_CV,axis=0))/np.std(X_CV,axis=0)

Y_CV=Y_CV.iloc[idx_real] 
P_f=np.sum(Y_CV==1)/len(Y_CV)
P_nf=1-P_f
# optimize SVM grid
if cross_val==True:

    print('Grid search hyperparameter optimisation started for SVM')
    t1=datetime.now()
    param_grid_svm={'class_weight':[{0:1e-2,1:1},{0:2e-2,1:1},{0:5e-2,1:1},\
                                    {0:1e-1,1:1}]}
    base_mdl_svm=SVC(kernel='linear',shrinking=False,\
                        probability=False,random_state=0,cache_size=1000,\
                            tol=X_CV.shape[-1]*1e-3)
    
    clf_svm_fk = GridSearchCV(base_mdl_svm, param_grid_svm,scoring='roc_auc',\
                        n_jobs=-1,cv=k_fold,refit=False)
    clf_svm_fk.fit(X_CV, Y_CV)
    opt_params_svm_fk=clf_svm_fk.best_params_
    C_opt=opt_params_svm_fk['class_weight'][0]
    score_svm=clf_svm_fk.best_score_
    
    t2=datetime.now()
    dt=t2-t1
    print('SVM CV finished after '+str(dt.total_seconds())+' sec')
    print('SVM: The optimal C+/C- ratio is '+str(1/C_opt))
else:
    opt_params_svm_fk={'class_weight':{0: 0.02, 1: 1}}
    score_svm=0.595534973722555
    print('Cross-validation skipped ... Using C+/C-='+\
          str(1/(opt_params_svm_fk['class_weight'][0])))


range_oos=range(start_OOS_year,end_OOS_year+1)
serial_fraud_count=np.zeros(len(range_oos))

# SVM based on Cecchini et al (2010) in MS
roc_svm=np.zeros(len(range_oos))
specificity_svm=np.zeros(len(range_oos))
sensitivity_OOS_svm=np.zeros(len(range_oos))
precision_svm=np.zeros(len(range_oos))
sensitivity_OOS_svm1=np.zeros(len(range_oos))
specificity_OOS_svm1=np.zeros(len(range_oos))
precision_svm1=np.zeros(len(range_oos))
ndcg_svm1=np.zeros(len(range_oos))
ecm_svm1=np.zeros(len(range_oos))

sensitivity_OOS_svm5=np.zeros(len(range_oos))
specificity_OOS_svm5=np.zeros(len(range_oos))
precision_svm5=np.zeros(len(range_oos))
ndcg_svm5=np.zeros(len(range_oos))
ecm_svm5=np.zeros(len(range_oos))
sensitivity_OOS_svm10=np.zeros(len(range_oos))
specificity_OOS_svm10=np.zeros(len(range_oos))
precision_svm10=np.zeros(len(range_oos))
ndcg_svm10=np.zeros(len(range_oos))
ecm_svm10=np.zeros(len(range_oos))

m=0

for yr in range_oos:
    t1=datetime.now()
    if case_window=='expanding':
        year_start_IS=1991
    else:
        year_start_IS=yr-IS_period
    idx_IS=tbl_ratio_fk[np.logical_and(tbl_ratio_fk.fyear<=yr-1,\
                                               tbl_ratio_fk.fyear>=year_start_IS)].index
    tbl_year_IS=tbl_ratio_fk.loc[idx_IS,:]
    misstate_firms=np.unique(tbl_year_IS.gvkey[tbl_year_IS.AAER_DUMMY==1])
    tbl_year_OOS=tbl_ratio_fk.loc[tbl_ratio_fk.fyear==yr]
    
    if adjust_serial==True:
        ok_index=np.zeros(tbl_year_OOS.shape[0])
        for s in range(0,tbl_year_OOS.shape[0]):
            if not tbl_year_OOS.iloc[s,1] in misstate_firms:
                ok_index[s]=True
        
    else:
        ok_index=np.ones(tbl_year_OOS.shape[0]).astype(bool)
        
    
    tbl_year_OOS=tbl_year_OOS.iloc[ok_index==True,:]

    X=mapped_X[idx_IS,:]
    idx_real=np.where(np.logical_and(np.isnan(X).any(axis=1)==False,\
                                     np.isinf(X).any(axis=1)==False))[0]
    X=X[idx_real,:]
    X=(X-np.mean(X,axis=0))/np.std(X,axis=0)
    Y=tbl_ratio_fk.AAER_DUMMY[idx_IS]
    Y=Y.iloc[idx_real]
    
    
    X_OOS=mapped_X[tbl_year_OOS.index,:]
    idx_real_OOS=np.where(np.logical_and(np.isnan(X_OOS).any(axis=1)==False,\
                                     np.isinf(X_OOS).any(axis=1)==False))[0]
    X_OOS=X_OOS[idx_real_OOS,:]
    X_OOS=(X_OOS-np.mean(X,axis=0))/np.std(X,axis=0)
    Y_OOS=tbl_year_OOS.AAER_DUMMY
    Y_OOS=Y_OOS.iloc[idx_real_OOS]
    Y_OOS=Y_OOS.reset_index(drop=True)
    
    n_P=np.sum(Y_OOS==1)
    n_N=np.sum(Y_OOS==0)
    
    
    # Support Vector Machines
    
    clf_svm_fk=SVC(class_weight=opt_params_svm_fk['class_weight'],kernel='linear',shrinking=False,\
                    probability=False,random_state=0,cache_size=1000,\
                        tol=X.shape[-1]*1e-3)
    clf_svm_fk=clf_svm_fk.fit(X,Y)
    pred_test_svm=clf_svm_fk.decision_function(X_OOS)
    pred_test_svm[pred_test_svm>=1]=1+np.log(pred_test_svm[pred_test_svm>=1])
    probs_oos_fraud_svm=np.exp(pred_test_svm)/(1+np.exp(pred_test_svm))
    

    labels_svm=clf_svm_fk.predict(X_OOS)
    
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
        
    cutoff_OOS_svm5=np.percentile(probs_oos_fraud_svm,95)
    sensitivity_OOS_svm5[m]=np.sum(np.logical_and(probs_oos_fraud_svm>=cutoff_OOS_svm5, \
                                                  Y_OOS==1))/np.sum(Y_OOS)
    specificity_OOS_svm5[m]=np.sum(np.logical_and(probs_oos_fraud_svm<cutoff_OOS_svm5, \
                                                  Y_OOS==0))/np.sum(Y_OOS==0)
    precision_svm5[m]=np.sum(np.logical_and(probs_oos_fraud_svm>=cutoff_OOS_svm5, \
                                                 Y_OOS==1))/np.sum(probs_oos_fraud_svm>=cutoff_OOS_svm5)
    ndcg_svm5[m]=ndcg_k(Y_OOS,probs_oos_fraud_svm,95)
    
    FN_svm5=np.sum(np.logical_and(probs_oos_fraud_svm<cutoff_OOS_svm5, \
                                                  Y_OOS==1))
    FP_svm5=np.sum(np.logical_and(probs_oos_fraud_svm>=cutoff_OOS_svm5, \
                                                  Y_OOS==0))
        
    ecm_svm5[m]=C_FN*P_f*FN_svm5/n_P+C_FP*P_nf*FP_svm5/n_N
    
    
    cutoff_OOS_svm10=np.percentile(probs_oos_fraud_svm,90)
    sensitivity_OOS_svm10[m]=np.sum(np.logical_and(probs_oos_fraud_svm>=cutoff_OOS_svm10, \
                                                  Y_OOS==1))/np.sum(Y_OOS)
    specificity_OOS_svm10[m]=np.sum(np.logical_and(probs_oos_fraud_svm<cutoff_OOS_svm10, \
                                                  Y_OOS==0))/np.sum(Y_OOS==0)
    precision_svm10[m]=np.sum(np.logical_and(probs_oos_fraud_svm>=cutoff_OOS_svm10, \
                                                 Y_OOS==1))/np.sum(probs_oos_fraud_svm>=cutoff_OOS_svm10)
    ndcg_svm10[m]=ndcg_k(Y_OOS,probs_oos_fraud_svm,90)
    
    FN_svm10=np.sum(np.logical_and(probs_oos_fraud_svm<cutoff_OOS_svm10, \
                                                  Y_OOS==1))
    FP_svm10=np.sum(np.logical_and(probs_oos_fraud_svm>=cutoff_OOS_svm10, \
                                                  Y_OOS==0))
        
    ecm_svm10[m]=C_FN*P_f*FN_svm10/n_P+C_FP*P_nf*FP_svm10/n_N
    
    t2=datetime.now() 
    dt=t2-t1
    print('analysis finished for OOS period '+str(yr)+' after '+str(dt.total_seconds())+' sec')
    m+=1

print('average top percentile sensitivity for the period '+str(start_OOS_year)+' to '+\
      str(end_OOS_year)+' is '+str(round(np.mean(sensitivity_OOS_svm1)*100,2))+\
          '% for SVM-FK')

# create performance table now
perf_tbl_general=pd.DataFrame()
perf_tbl_general['models']=['SVM-FK-23']

perf_tbl_general['Roc']=np.mean(roc_svm)
                                            
perf_tbl_general['Sensitivity @ 1 Prc']=np.mean(sensitivity_OOS_svm1)

perf_tbl_general['Specificity @ 1 Prc']=np.mean(specificity_OOS_svm1)

perf_tbl_general['Precision @ 1 Prc']=np.mean(precision_svm1)

perf_tbl_general['F1 Score @ 1 Prc']=2*(perf_tbl_general['Precision @ 1 Prc']*\
                                      perf_tbl_general['Sensitivity @ 1 Prc'])/\
                                        ((perf_tbl_general['Precision @ 1 Prc']+\
                                          perf_tbl_general['Sensitivity @ 1 Prc']))
                                            
perf_tbl_general['NDCG @ 1 Prc']=np.mean(ndcg_svm1)

perf_tbl_general['ECM @ 1 Prc']=np.mean(ecm_svm1)

perf_tbl_general['Sensitivity @ 5 Prc']=np.mean(sensitivity_OOS_svm5)

perf_tbl_general['Specificity @ 5 Prc']=np.mean(specificity_OOS_svm5)

perf_tbl_general['Precision @ 5 Prc']=np.mean(precision_svm5)

perf_tbl_general['F1 Score @ 5 Prc']=2*(perf_tbl_general['Precision @ 5 Prc']*\
                                      perf_tbl_general['Sensitivity @ 5 Prc'])/\
                                        ((perf_tbl_general['Precision @ 5 Prc']+\
                                          perf_tbl_general['Sensitivity @ 5 Prc']))
                                            
perf_tbl_general['NDCG @ 5 Prc']=np.mean(ndcg_svm5)

perf_tbl_general['ECM @ 5 Prc']=np.mean(ecm_svm5)

perf_tbl_general['Sensitivity @ 10 Prc']=np.mean(sensitivity_OOS_svm10)


perf_tbl_general['Specificity @ 10 Prc']=np.mean(specificity_OOS_svm10)
    
perf_tbl_general['Precision @ 10 Prc']=np.mean(precision_svm10)

perf_tbl_general['F1 Score @ 10 Prc']=2*(perf_tbl_general['Precision @ 10 Prc']*\
                                      perf_tbl_general['Sensitivity @ 10 Prc'])/\
                                        ((perf_tbl_general['Precision @ 10 Prc']+\
                                          perf_tbl_general['Sensitivity @ 10 Prc']))
                                            
perf_tbl_general['NDCG @ 10 Prc']=np.mean(ndcg_svm10)  
perf_tbl_general['ECM @ 10 Prc']=np.mean(ecm_svm10)

perf_tbl_general['Sensitivity']=np.mean(sensitivity_OOS_svm)
    
perf_tbl_general['Specificity']=np.mean(specificity_svm)

perf_tbl_general['Precision']=np.mean(precision_svm)

perf_tbl_general['F1 Score']=2*(perf_tbl_general['Precision']*\
                                      perf_tbl_general['Sensitivity'])/\
                                        ((perf_tbl_general['Precision']+\
                                          perf_tbl_general['Sensitivity']))                                             

if case_window=='expanding':
    lbl_perf_tbl='perf_tbl_'+str(start_OOS_year)+'_'+str(end_OOS_year)+\
        '_'+case_window+',OOS='+str(OOS_period)+','+\
        str(k_fold)+'fold'+',serial='+str(adjust_serial)+'_SVM_FK.csv'
else:
    lbl_perf_tbl='perf_tbl_'+str(start_OOS_year)+'_'+str(end_OOS_year)+\
        '_IS='+str(IS_period)+',OOS='+str(OOS_period)+','+\
        str(k_fold)+'fold'+',serial='+str(adjust_serial)+'_SVM_FK.csv'

perf_tbl_general.to_csv(lbl_perf_tbl,index=False)
print(perf_tbl_general)
t_last=datetime.now()
dt_total=t_last-t0
print('total run time is '+str(dt_total.total_seconds())+' sec')


# extract performance for 2003-2008 directly from 2001-2010

perf_tbl_general=pd.DataFrame()
perf_tbl_general['models']=['SVM-FK-23']

perf_tbl_general['Roc']=np.mean(roc_svm[2:8])
                                            
perf_tbl_general['Sensitivity @ 1 Prc']=np.mean(sensitivity_OOS_svm1[2:8])

perf_tbl_general['Specificity @ 1 Prc']=np.mean(specificity_OOS_svm1[2:8])

perf_tbl_general['Precision @ 1 Prc']=np.mean(precision_svm1[2:8])

perf_tbl_general['F1 Score @ 1 Prc']=2*(perf_tbl_general['Precision @ 1 Prc']*\
                                      perf_tbl_general['Sensitivity @ 1 Prc'])/\
                                        ((perf_tbl_general['Precision @ 1 Prc']+\
                                          perf_tbl_general['Sensitivity @ 1 Prc']))
                                            
perf_tbl_general['NDCG @ 1 Prc']=np.mean(ndcg_svm1[2:8])

perf_tbl_general['ECM @ 1 Prc']=np.mean(ecm_svm1[2:8])

perf_tbl_general['Sensitivity @ 5 Prc']=np.mean(sensitivity_OOS_svm5[2:8])

perf_tbl_general['Specificity @ 5 Prc']=np.mean(specificity_OOS_svm5[2:8])

perf_tbl_general['Precision @ 5 Prc']=np.mean(precision_svm5[2:8])

perf_tbl_general['F1 Score @ 5 Prc']=2*(perf_tbl_general['Precision @ 5 Prc']*\
                                      perf_tbl_general['Sensitivity @ 5 Prc'])/\
                                        ((perf_tbl_general['Precision @ 5 Prc']+\
                                          perf_tbl_general['Sensitivity @ 5 Prc']))
                                            
perf_tbl_general['NDCG @ 5 Prc']=np.mean(ndcg_svm5[2:8])

perf_tbl_general['ECM @ 5 Prc']=np.mean(ecm_svm5[2:8])

perf_tbl_general['Sensitivity @ 10 Prc']=np.mean(sensitivity_OOS_svm10[2:8])


perf_tbl_general['Specificity @ 10 Prc']=np.mean(specificity_OOS_svm10[2:8])
    
perf_tbl_general['Precision @ 10 Prc']=np.mean(precision_svm10[2:8])

perf_tbl_general['F1 Score @ 10 Prc']=2*(perf_tbl_general['Precision @ 10 Prc']*\
                                      perf_tbl_general['Sensitivity @ 10 Prc'])/\
                                        ((perf_tbl_general['Precision @ 10 Prc']+\
                                          perf_tbl_general['Sensitivity @ 10 Prc']))
                                            
perf_tbl_general['NDCG @ 10 Prc']=np.mean(ndcg_svm10[2:8])  

perf_tbl_general['ECM @ 10 Prc']=np.mean(ecm_svm10[2:8])

perf_tbl_general['Sensitivity']=np.mean(sensitivity_OOS_svm[2:8])
    
perf_tbl_general['Specificity']=np.mean(specificity_svm[2:8])

perf_tbl_general['Precision']=np.mean(precision_svm[2:8])

perf_tbl_general['F1 Score']=2*(perf_tbl_general['Precision']*\
                                      perf_tbl_general['Sensitivity'])/\
                                        ((perf_tbl_general['Precision']+\
                                          perf_tbl_general['Sensitivity']))                                             

if case_window=='expanding':
    lbl_perf_tbl='perf_tbl_'+str(2003)+'_'+str(2008)+\
        '_'+case_window+',OOS='+str(OOS_period)+','+\
        str(k_fold)+'fold'+',serial='+str(adjust_serial)+'_SVM_FK.csv'
else:
    lbl_perf_tbl='perf_tbl_'+str(2003)+'_'+str(2008)+\
        '_IS='+str(IS_period)+',OOS='+str(OOS_period)+','+\
        str(k_fold)+'fold'+',serial='+str(adjust_serial)+'_SVM_FK.csv'
perf_tbl_general.to_csv(lbl_perf_tbl,index=False)
