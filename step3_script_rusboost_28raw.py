#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 12:24:27 2021

This code replicates the RUSBoost model of Bao et al (2020)
https://doi.org/10.1111/1475-679X.12292
We use the 28 raw financial variables as in Bao et al (2020)

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
    – RUSBoost based on Scikit module
Outputs: 
Main results are stored in the table variable "perf_tbl_general" written into
2 csv files: time period 2001-2010 and 2003-2008

Steps:
    1. Estimating the performance for each OOS period.

Warnings: 
    – Running this code can take up to 180 mins. 
    These figures are estimates based on a MacBook Pro 2017.

@author: Arman Hassanniakalager GitHub: https://github.com/hkalager
Common disclaimers apply. Subject to change at all time.

Last review: 21/07/2021
"""


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
import rusboost
from sklearn.metrics import roc_auc_score
from extra_codes import ndcg_k



base_tree=DecisionTreeClassifier(min_samples_leaf=5)
bao_RUSboost=rusboost.RUSBoost(base_estimator=base_tree,n_estimators=3000,\
                             learning_rate=.1,min_ratio=1,random_state=0)
t0=datetime.now()

OOS_period=1
start_OOS_year=2001
end_OOS_year=2010
adjust_serial=False
fraud_df=pd.read_csv('FraudDB2020.csv')

reduced_tbl_1=fraud_df.iloc[:,[0,1,3,7,8]]
reduced_tbl_2=fraud_df.iloc[:,9:-14]
reduced_tblset=[reduced_tbl_1,reduced_tbl_2]
reduced_tbl=pd.concat(reduced_tblset,axis=1)
reduced_tbl=reduced_tbl.reset_index(drop=True)

range_oos=range(start_OOS_year,end_OOS_year+1)


roc_rusboost=np.zeros(len(range_oos))
specificity_rusboost=np.zeros(len(range_oos))
sensitivity_OOS_rusboost=np.zeros(len(range_oos))
precision_rusboost=np.zeros(len(range_oos))
sensitivity_OOS_rusboost1=np.zeros(len(range_oos))
specificity_OOS_rusboost1=np.zeros(len(range_oos))
precision_rusboost1=np.zeros(len(range_oos))
ndcg_rusboost1=np.zeros(len(range_oos))
sensitivity_OOS_rusboost5=np.zeros(len(range_oos))
specificity_OOS_rusboost5=np.zeros(len(range_oos))
precision_rusboost5=np.zeros(len(range_oos))
ndcg_rusboost5=np.zeros(len(range_oos))
sensitivity_OOS_rusboost10=np.zeros(len(range_oos))
specificity_OOS_rusboost10=np.zeros(len(range_oos))
precision_rusboost10=np.zeros(len(range_oos))
ndcg_rusboost10=np.zeros(len(range_oos))

m=0

for yr in range_oos:
    t1=datetime.now()
    tbl_year_IS=reduced_tbl.loc[np.logical_and(reduced_tbl.fyear<=yr-1,\
                                               reduced_tbl.fyear>=1991)]
    tbl_year_IS=tbl_year_IS.reset_index(drop=True)
    misstate_firms=np.unique(tbl_year_IS.gvkey[tbl_year_IS.AAER_DUMMY==1])
    tbl_year_OOS=reduced_tbl.loc[reduced_tbl.fyear==yr]
    
    if adjust_serial==True:
        ok_index=np.zeros(tbl_year_OOS.shape[0])
        for s in range(0,tbl_year_OOS.shape[0]):
            if not tbl_year_OOS.iloc[s,1] in misstate_firms:
                ok_index[s]=True
        
    else:
        ok_index=np.ones(tbl_year_OOS.shape[0]).astype(bool)
        
    
    tbl_year_OOS=tbl_year_OOS.iloc[ok_index==True,:]
    tbl_year_OOS=tbl_year_OOS.reset_index(drop=True)
    
    X=tbl_year_IS.iloc[:,-28:]
    
    Y=tbl_year_IS.AAER_DUMMY
    
    X_OOS=tbl_year_OOS.iloc[:,-28:]
    #X_OOS=(X_OOS-mean_vals)/std_vals
    
    Y_OOS=tbl_year_OOS.AAER_DUMMY
    
    clf_rusboost = bao_RUSboost.fit(X,Y)
    
    probs_oos_fraud_rusboost=clf_rusboost.predict_proba(X_OOS)[:,-1]
    
    labels_rusboost=clf_rusboost.predict(X_OOS)
    
    roc_rusboost[m]=roc_auc_score(Y_OOS,probs_oos_fraud_rusboost)
    specificity_rusboost[m]=np.sum(np.logical_and(labels_rusboost==0,Y_OOS==0))/\
        np.sum(Y_OOS==0)
    
    sensitivity_OOS_rusboost[m]=np.sum(np.logical_and(labels_rusboost==1, \
                                                 Y_OOS==1))/np.sum(Y_OOS)
    precision_rusboost[m]=np.sum(np.logical_and(labels_rusboost==1,Y_OOS==1))/np.sum(labels_rusboost)
    
    
    cutoff_OOS_rusboost=np.percentile(probs_oos_fraud_rusboost,99)
    sensitivity_OOS_rusboost1[m]=np.sum(np.logical_and(probs_oos_fraud_rusboost>=cutoff_OOS_rusboost, \
                                                 Y_OOS==1))/np.sum(Y_OOS)
    specificity_OOS_rusboost1[m]=np.sum(np.logical_and(probs_oos_fraud_rusboost<cutoff_OOS_rusboost, \
                                                  Y_OOS==0))/np.sum(Y_OOS==0)
    precision_rusboost1[m]=np.sum(np.logical_and(probs_oos_fraud_rusboost>=cutoff_OOS_rusboost, \
                                                 Y_OOS==1))/np.sum(probs_oos_fraud_rusboost>=cutoff_OOS_rusboost)
    ndcg_rusboost1[m]=ndcg_k(Y_OOS,probs_oos_fraud_rusboost,99)
    
    cutoff_OOS_rusboost5=np.percentile(probs_oos_fraud_rusboost,95)
    sensitivity_OOS_rusboost5[m]=np.sum(np.logical_and(probs_oos_fraud_rusboost>=cutoff_OOS_rusboost5, \
                                                 Y_OOS==1))/np.sum(Y_OOS)
    specificity_OOS_rusboost5[m]=np.sum(np.logical_and(probs_oos_fraud_rusboost<cutoff_OOS_rusboost5, \
                                                  Y_OOS==0))/np.sum(Y_OOS==0)
    precision_rusboost5[m]=np.sum(np.logical_and(probs_oos_fraud_rusboost>=cutoff_OOS_rusboost5, \
                                                 Y_OOS==1))/np.sum(probs_oos_fraud_rusboost>=cutoff_OOS_rusboost5)
    ndcg_rusboost5[m]=ndcg_k(Y_OOS,probs_oos_fraud_rusboost,95)
    
    cutoff_OOS_rusboost10=np.percentile(probs_oos_fraud_rusboost,90)
    sensitivity_OOS_rusboost10[m]=np.sum(np.logical_and(probs_oos_fraud_rusboost>=cutoff_OOS_rusboost10, \
                                                 Y_OOS==1))/np.sum(Y_OOS)
    specificity_OOS_rusboost10[m]=np.sum(np.logical_and(probs_oos_fraud_rusboost<cutoff_OOS_rusboost10, \
                                                  Y_OOS==0))/np.sum(Y_OOS==0)
    precision_rusboost10[m]=np.sum(np.logical_and(probs_oos_fraud_rusboost>=cutoff_OOS_rusboost10, \
                                                 Y_OOS==1))/np.sum(probs_oos_fraud_rusboost>=cutoff_OOS_rusboost10)
    ndcg_rusboost10[m]=ndcg_k(Y_OOS,probs_oos_fraud_rusboost,90)
    

    
    t2=datetime.now() 
    dt=t2-t1
    print('analysis finished for OOS period '+str(yr)+' after '+str(dt.total_seconds())+' sec')
    m+=1

print('average top percentile sensitivity for the period '+str(start_OOS_year)+' to '+\
      str(end_OOS_year)+' is '+str(round(np.mean(sensitivity_OOS_rusboost1)*100,2))+\
          '% for RUSBoost-28')

# create performance table now
perf_tbl_general=pd.DataFrame()
perf_tbl_general['models']=['RUSBoost-28']

perf_tbl_general['Roc']=np.mean(roc_rusboost)
                                            
perf_tbl_general['Sensitivity @ 1 Prc']=np.mean(sensitivity_OOS_rusboost1)

perf_tbl_general['Specificity @ 1 Prc']=np.mean(specificity_OOS_rusboost1)

perf_tbl_general['Precision @ 1 Prc']=np.mean(precision_rusboost1)

perf_tbl_general['F1 Score @ 1 Prc']=2*(perf_tbl_general['Precision @ 1 Prc']*\
                                      perf_tbl_general['Sensitivity @ 1 Prc'])/\
                                        ((perf_tbl_general['Precision @ 1 Prc']+\
                                          perf_tbl_general['Sensitivity @ 1 Prc']))
                                            
perf_tbl_general['NDCG @ 1 Prc']=np.mean(ndcg_rusboost1)

perf_tbl_general['Sensitivity @ 5 Prc']=np.mean(sensitivity_OOS_rusboost5)

perf_tbl_general['Specificity @ 5 Prc']=np.mean(specificity_OOS_rusboost5)

perf_tbl_general['Precision @ 5 Prc']=np.mean(precision_rusboost5)

perf_tbl_general['F1 Score @ 5 Prc']=2*(perf_tbl_general['Precision @ 5 Prc']*\
                                      perf_tbl_general['Sensitivity @ 5 Prc'])/\
                                        ((perf_tbl_general['Precision @ 5 Prc']+\
                                          perf_tbl_general['Sensitivity @ 5 Prc']))
                                            
perf_tbl_general['NDCG @ 5 Prc']=np.mean(ndcg_rusboost5)

perf_tbl_general['Sensitivity @ 10 Prc']=np.mean(sensitivity_OOS_rusboost10)


perf_tbl_general['Specificity @ 10 Prc']=np.mean(specificity_OOS_rusboost10)
    
perf_tbl_general['Precision @ 10 Prc']=np.mean(precision_rusboost10)

perf_tbl_general['F1 Score @ 10 Prc']=2*(perf_tbl_general['Precision @ 10 Prc']*\
                                      perf_tbl_general['Sensitivity @ 10 Prc'])/\
                                        ((perf_tbl_general['Precision @ 10 Prc']+\
                                          perf_tbl_general['Sensitivity @ 10 Prc']))
                                            
perf_tbl_general['NDCG @ 10 Prc']=np.mean(ndcg_rusboost10)  

perf_tbl_general['Sensitivity']=np.mean(sensitivity_OOS_rusboost)
    
perf_tbl_general['Specificity']=np.mean(specificity_rusboost)

perf_tbl_general['Precision']=np.mean(precision_rusboost)

perf_tbl_general['F1 Score']=2*(perf_tbl_general['Precision']*\
                                      perf_tbl_general['Sensitivity'])/\
                                        ((perf_tbl_general['Precision']+\
                                          perf_tbl_general['Sensitivity']))                                             

lbl_perf_tbl='perf_tbl_rusboost_'+str(start_OOS_year)+'_'+str(end_OOS_year)+\
                '_expanding_OOS='+str(OOS_period)+',serial='+str(adjust_serial)+'.csv'
perf_tbl_general.to_csv(lbl_perf_tbl,index=False)
print(perf_tbl_general)
t_last=datetime.now()
dt_total=t_last-t0
print('total run time is '+str(dt_total.total_seconds())+' sec')

# extract performance for 2003-2008 directly from 2001-2010

perf_tbl_general=pd.DataFrame()
perf_tbl_general['models']=['RUSBoost-28']

perf_tbl_general['Roc']=np.mean(roc_rusboost[2:7])
                                            
perf_tbl_general['Sensitivity @ 1 Prc']=np.mean(sensitivity_OOS_rusboost1[2:7])

perf_tbl_general['Specificity @ 1 Prc']=np.mean(specificity_OOS_rusboost1[2:7])

perf_tbl_general['Precision @ 1 Prc']=np.mean(precision_rusboost1[2:7])

perf_tbl_general['F1 Score @ 1 Prc']=2*(perf_tbl_general['Precision @ 1 Prc']*\
                                      perf_tbl_general['Sensitivity @ 1 Prc'])/\
                                        ((perf_tbl_general['Precision @ 1 Prc']+\
                                          perf_tbl_general['Sensitivity @ 1 Prc']))
                                            
perf_tbl_general['NDCG @ 1 Prc']=np.mean(ndcg_rusboost1[2:7])

perf_tbl_general['Sensitivity @ 5 Prc']=np.mean(sensitivity_OOS_rusboost5[2:7])

perf_tbl_general['Specificity @ 5 Prc']=np.mean(specificity_OOS_rusboost5[2:7])

perf_tbl_general['Precision @ 5 Prc']=np.mean(precision_rusboost5[2:7])

perf_tbl_general['F1 Score @ 5 Prc']=2*(perf_tbl_general['Precision @ 5 Prc']*\
                                      perf_tbl_general['Sensitivity @ 5 Prc'])/\
                                        ((perf_tbl_general['Precision @ 5 Prc']+\
                                          perf_tbl_general['Sensitivity @ 5 Prc']))
                                            
perf_tbl_general['NDCG @ 5 Prc']=np.mean(ndcg_rusboost5[2:7])

perf_tbl_general['Sensitivity @ 10 Prc']=np.mean(sensitivity_OOS_rusboost10[2:7])


perf_tbl_general['Specificity @ 10 Prc']=np.mean(specificity_OOS_rusboost10[2:7])
    
perf_tbl_general['Precision @ 10 Prc']=np.mean(precision_rusboost10[2:7])

perf_tbl_general['F1 Score @ 10 Prc']=2*(perf_tbl_general['Precision @ 10 Prc']*\
                                      perf_tbl_general['Sensitivity @ 10 Prc'])/\
                                        ((perf_tbl_general['Precision @ 10 Prc']+\
                                          perf_tbl_general['Sensitivity @ 10 Prc']))
                                            
perf_tbl_general['NDCG @ 10 Prc']=np.mean(ndcg_rusboost10[2:7])  

perf_tbl_general['Sensitivity']=np.mean(sensitivity_OOS_rusboost[2:7])
    
perf_tbl_general['Specificity']=np.mean(specificity_rusboost[2:7])

perf_tbl_general['Precision']=np.mean(precision_rusboost[2:7])

perf_tbl_general['F1 Score']=2*(perf_tbl_general['Precision']*\
                                      perf_tbl_general['Sensitivity'])/\
                                        ((perf_tbl_general['Precision']+\
                                          perf_tbl_general['Sensitivity']))                                             

lbl_perf_tbl='perf_tbl_rusboost_'+str(2003)+'_'+str(2008)+\
                '_expanding_OOS='+str(OOS_period)+',serial='+str(adjust_serial)+'.csv'
perf_tbl_general.to_csv(lbl_perf_tbl,index=False)
