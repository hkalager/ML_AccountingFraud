#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module created on Tue Jun 21 13:49:12 2022

The module requires either a DataFrame titled "FraudDB2020.csv" or 
4x smaller slices of it. The sample csv file is accompanied.

The csv file "FraudDB2020.csv" must include information regarding 
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


@author: Arman Hassanniakalager GitHub: https://github.com/hkalager
Common disclaimers apply. Subject to change at all time.

Last review: 20/11/2022
"""
# %matplotlib inline
import pandas as pd
import numpy as np
from os.path import isfile
from extra_codes import calc_vif
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import kpss
import warnings
warnings.filterwarnings("ignore")

class ML_Fraud:
    __version__='1.0.5'
    def __init__(self,sample_start=1991,test_sample=range(2001,2011),
                 OOS_per=1,OOS_gap=0,sampling='expanding',adjust_serial=True,
                 cv_type='kfold',temp_year=1,cv_flag=False,cv_k=10,write=True,IS_per=10):
        """
        Parameters:
            – sample_start: Calendar year marking the start of the sample (Default=1991)
            – test_sample: testing/out-of-sample period(Default=range(2001,2011))
            – OOS_per: out-of-sample rolling period in years (Default=OOS_per=1)
            – OOS_gap: Gap between training and testing samples in year (Default=0)
            – sampling: sampling style either "expanding"/"rolling" (Default="expanding")
            – adjust_serial: A boolean variable to adjust for serial frauds (Default=True)
            – cv_type: A string to determine whether to do a temporal or k-fold cv
            – cv_flag: A boolean variable whether to replicate the cross-validation (Default=False)
            – cv_k: The number of folds (k) in the cross-validation (Default=10)
            – write: A boolean variable whether to write results into csv files (Default=True)
            – IS_per: Number of calendar years in case a rolling training sample is used (Default=10)
            
        """
        if isfile('FraudDB2020.csv')==False:
            df=pd.DataFrame()
            for s in range(1,5):
                fl_name='FraudDB2020_Part'+str(s)+'.csv'
                new_df=pd.read_csv(fl_name)
                df=df.append(new_df)
            df.to_csv('FraudDB2020.csv',index=False)
            
        df=pd.read_csv('FraudDB2020.csv')
        self.df=df
        self.ss=sample_start
        self.se=np.max(df.fyear)
        self.ts=test_sample
        self.cv_t=cv_type
        self.cv=cv_flag
        self.cv_k=cv_k
        self.cv_t_y=temp_year
        
        sampling_set=['expanding','rolling']
        if sampling in sampling_set:
            pass
        else:
            raise ValueError('Invalid sampling choice. Permitted options are "expanding" and "rolling"')
        
        self.sa=sampling
        self.w=write
        self.ip=IS_per
        self.op=OOS_per
        self.og=OOS_gap
        self.a_s=adjust_serial
        print('Module initiated successfully ...')
        list_methods=dir(self)
        reduced_methods=[item+'()' for item in list_methods if any(['analy' in item,'compare' in item,item=='sumstats'])]
        print('Procedures are: '+'; '.join(reduced_methods))
    
    def sumstats(self):
        """
        Generate summary statistics and VIF results for the three set of predictors.
        
        Warning:
            – Running this code can take up to 1 min. 
            These figures are estimates based on a MacBook Pro 2021

        """
        
        t0=datetime.now()
        
        fraud_df=self.df
        sample_start=self.ss
        sample_end=self.se
        end_OOS_year=self.ts[-1]
        IS_per=self.ip
        write=self.w

        fyears_available=[s for s in range(sample_start,sample_end+1)]

        count_fraud=[sum(np.logical_and(fraud_df.fyear==yr,fraud_df.AAER_DUMMY==1)) for 
                     yr in fyears_available]


        count_firms=[np.unique(fraud_df.gvkey[fraud_df.fyear==yr]).shape[0] for 
                     yr in fyears_available]

        basic_table=pd.DataFrame()
        basic_table['Year']=fyears_available
        basic_table['Num_fraud']=count_fraud
        basic_table['Num_firm']=count_firms
        basic_table['Fraud/Firm']=basic_table['Num_fraud']/basic_table['Num_firm']

        print(' Average number of new frauds per year is '+str(basic_table['Num_fraud'].mean()))
        print(' Average number of unique firms per year is '+str(basic_table['Num_firm'].mean()))

        kpss_test_fraud=kpss(basic_table['Fraud/Firm'],regression='c', nlags='auto')
        if kpss_test_fraud[1]<0.1:
            print('Stationarity rejected ')
        else:
            print('# fraud/#unique firm is stationary over time ...')

        fraud_df=fraud_df[fraud_df.fyear>=sample_start]
        last_year=np.max(fraud_df.fyear)
        fraud_df=fraud_df.reset_index(drop=True)
        num_comp=len(np.unique(fraud_df.gvkey))
        print(str(num_comp)+' unique firms in the dataset between '+str(sample_start)+' and '+str(last_year))


        reduced_tbl_ratio=fraud_df.iloc[:,-14:-3]
        reduced_tbl_ratio=(reduced_tbl_ratio-np.mean(reduced_tbl_ratio))/np.std(reduced_tbl_ratio)
        vif_ratios=calc_vif(reduced_tbl_ratio)
        if write==True:
            vif_ratios.to_csv('VIF_11ratio.csv',index=False)

        reduced_tbl_raw28=fraud_df.iloc[:,9:-14]
        reduced_tbl_raw28=(reduced_tbl_raw28-np.mean(reduced_tbl_raw28))/np.std(reduced_tbl_raw28)
        vif_raw28=calc_vif(reduced_tbl_raw28)
        if write==True:
            vif_raw28.to_csv('VIF_28raw.csv',index=False)


        reduced_tbl_raw23=fraud_df.iloc[:,9:-14]
        reduced_tbl_raw23.pop('act')
        reduced_tbl_raw23.pop('ap')
        reduced_tbl_raw23.pop('ppegt')
        reduced_tbl_raw23.pop('dltis')
        reduced_tbl_raw23.pop('sstk')
        reduced_tbl_raw23=(reduced_tbl_raw23-np.mean(reduced_tbl_raw23))/np.std(reduced_tbl_raw23)
        vif_raw23=calc_vif(reduced_tbl_raw23)
        if write==True:
            vif_raw23.to_csv('VIF_23raw.csv',index=False)
        print('VIF results generated successfully ... ')

        sum_Stat_tbl=pd.DataFrame()
        itr=0
        for sel_column in reduced_tbl_ratio.columns:
            sel_data=reduced_tbl_ratio[sel_column]
            sum_Stat_tbl.loc[itr,'variable']=sel_column
            sum_Stat_tbl.loc[itr,'min']=round(np.min(sel_data),4)
            sum_Stat_tbl.loc[itr,'max']=round(np.max(sel_data),4)
            sum_Stat_tbl.loc[itr,'mean']=round(np.mean(sel_data),4)
            sum_Stat_tbl.loc[itr,'std']=round(np.std(sel_data),4)
            itr+=1

        for sel_column in reduced_tbl_raw28.columns:
            sel_data=reduced_tbl_raw28[sel_column]
            sum_Stat_tbl.loc[itr,'variable']=sel_column
            sum_Stat_tbl.loc[itr,'min']=round(np.min(sel_data),4)
            sum_Stat_tbl.loc[itr,'max']=round(np.max(sel_data),4)
            sum_Stat_tbl.loc[itr,'mean']=round(np.mean(sel_data),4)
            sum_Stat_tbl.loc[itr,'std']=round(np.std(sel_data),4)
            itr+=1
            
        if write==True:
            sum_Stat_tbl.to_csv('SumStats.csv',index=False)
            print('Summary statistics generated successfully ... ')

        

        range_sample=range(sample_start,sample_end+1)
        serial_fraud_count=np.zeros(len(range_sample))
        fraud_count=np.zeros(len(range_sample))
        m=0

        for yr in range_sample:
            tbl_year_IS=fraud_df.loc[np.logical_and(fraud_df.fyear<yr,\
                                                       fraud_df.fyear>=yr-IS_per)]
            tbl_year_IS=tbl_year_IS.reset_index(drop=True)
            misstate_firms=np.unique(tbl_year_IS.gvkey[tbl_year_IS.AAER_DUMMY==1])
            tbl_year_OOS=fraud_df[fraud_df.fyear==yr]
            tbl_year_OOS=tbl_year_OOS.reset_index(drop=True)
            fraud_count[m]=np.sum(tbl_year_OOS.AAER_DUMMY.values)
            serial_fraud_count[m]=0
            for s in range(0,tbl_year_OOS.shape[0]):
                serial_case=np.logical_and((tbl_year_OOS.iloc[s,1] in misstate_firms),\
                                           tbl_year_OOS.AAER_DUMMY[s]==1).astype(int)
                serial_fraud_count[m]=serial_fraud_count[m]+serial_case
                 
            m+=1


        ser_tbl=pd.DataFrame()
        ser_tbl['Year']=range_sample
        ser_tbl['Fraud Case']=fraud_count
        ser_tbl['Serial Fraud Case']=serial_fraud_count
        ser_tbl['Serial over Total']=serial_fraud_count/fraud_count

        if write==True:
            ser_tbl.to_csv('SerialStats.csv',index=False)
            print('Serial fraud results generated successfully ...')
        
        fig, ax = plt.subplots()
        X_axis=pd.to_datetime(fyears_available,format='%Y')
        
        ax.plot(X_axis,serial_fraud_count,'s:r',label='serial fraud')
        ax.plot(X_axis,fraud_count,'s:b',label='total fraud')
        ax.set_xlabel('Calendar Year')
        ax.set_ylabel('# AAERs')
        ax.set_title('unique firms and AAERs over time')
        ax2=ax.twinx()
        ax2.plot(X_axis,count_firms,'^--g',label='unique firm')
        ax2.set_ylabel('# firms')
        ax.legend()
        ax2.legend(loc=2)
        plt.savefig('Figure1.png')
        plt.show()
        
        print('graphics generated successfully ... ')
        run_time=datetime.now()-t0
        print('Total runtime is '+str(run_time.total_seconds())+' seconds')
        ## End of Summary Statistics procedure 
      
    def mc_analysis(self,B=1000,adjust_serial=None):
        """
        This method runs a Monte Carlo simulation to quantify the impact of
        serial frauds on ratios vs raw variables datasets.
        
        Parameters:
            – B: Total number of random boostrap replications 
            – adjust_serial: treatment for serial frauds as choices of 
            True/False/ 'biased' where True discards spanning AAER cases in the 
            testing sample. False ignores the issue, and finally 'biased' treats
            the serial frauds as done in Bao et al (2020)

        Outputs: 
        Main results are stored in the table variable "result_tbls" written into
        a csv file per treatment type named: 
            'MC_results'+',serial='+str(adjust_serial)+'.csv'
        
            
        """
        
        from sklearn.metrics import roc_auc_score
        from extra_codes import ndcg_k,relogit
        from statsmodels.discrete.discrete_model import Logit
        from statsmodels.tools import add_constant
        
        t0=datetime.now()
        # setting the parameters
        IS_period=self.ip
        k_fold=self.cv_k
        OOS_period=self.op # 1 year ahead prediction
        OOS_gap=self.og # Gap between training and testing period
        start_OOS_year=self.ts[0]
        end_OOS_year=self.ts[-1]
        sample_start=self.ss
        if adjust_serial==None:
            adjust_serial=self.a_s
        cross_val=self.cv
        case_window=self.sa
        fraud_df=self.df
        write=self.w
        
        
        print('starting the MC analysis for case B='+str(B)+', serial treatment='+str(adjust_serial))
        t000=datetime.now()
        reduced_tbl_1=fraud_df.iloc[:,[0,1,3,7,8]]
        reduced_tbl_2=fraud_df.iloc[:,9:-3]
        reduced_tblset=[reduced_tbl_1,reduced_tbl_2]
        reduced_tbl=pd.concat(reduced_tblset,axis=1)
        reduced_tbl=reduced_tbl[reduced_tbl.fyear>=sample_start]
        reduced_tbl=reduced_tbl[reduced_tbl.fyear<=end_OOS_year]

        # Setting the cross-validation setting

        tbl_year_IS_CV=reduced_tbl.loc[np.logical_and(reduced_tbl.fyear<=2010,\
                                                   reduced_tbl.fyear>=1991)]
        tbl_year_IS_CV=tbl_year_IS_CV.reset_index(drop=True)
        
        X_CV=tbl_year_IS_CV.iloc[:,-11:]
        
        Y_CV=tbl_year_IS_CV.AAER_DUMMY
        
        
        X_CV_raw=tbl_year_IS_CV.iloc[:,5:-11]
        
        
        
        idx_set=tbl_year_IS_CV.index
        cv_size=X_CV.shape[0]
        count_train=int(((k_fold-1)/k_fold)*cv_size)
        
        
        roc_set_ratio=[]
        count_positive_train=[]
        count_positive_test=[]
        drop_serial=[]
        ndcg_ratio=[]
        sensitivity_ratio=[]
        specificity_ratio=[]
        precision_ratio=[]
        roc_set_raw=[]
        sensitivity_raw=[]
        specificity_raw=[]
        precision_raw=[]
        ndcg_raw=[]
        t0=datetime.now()
        for b in range(0,B):
            t00=datetime.now()
            rng = np.random.default_rng(b)
            train_idx=rng.choice(idx_set,count_train,replace=False)
            #train_idx=np.random.choice(idx_set,count_train,replace=False,)            
            test_idx=idx_set.drop(train_idx)
            mini_tbl_train=tbl_year_IS_CV.iloc[train_idx]
            mini_tbl_test=tbl_year_IS_CV.iloc[test_idx]
            #test_idx=rng.choice(idx_set_test,size_test,replace=True)
            
            X_train_b=X_CV.iloc[train_idx,:]
            X_train_raw_b=X_CV_raw.iloc[train_idx,:]
            Y_train_b=Y_CV.iloc[train_idx]
            count_positive_train.append(np.sum(Y_train_b))
            X_test_b=X_CV.iloc[test_idx,:]
            X_test_raw_b=X_CV_raw.iloc[test_idx,:]
            Y_test_b=Y_CV.iloc[test_idx]
            init_positive=np.sum(Y_test_b)
            count_positive_test.append(init_positive)
            if adjust_serial==True:
                misstate_firms_train=np.unique(mini_tbl_train[mini_tbl_train.AAER_DUMMY==1]['gvkey'])
                
                idx_is_serial=np.isin(mini_tbl_test['gvkey'],misstate_firms_train)
                X_test_b=X_test_b[idx_is_serial==False]
                X_test_raw_b=X_test_raw_b[idx_is_serial==False]
                Y_test_b=Y_test_b[idx_is_serial==False]
                                
                adj_positive=np.sum(Y_test_b)
                drop_serial.append(init_positive-adj_positive)
            elif adjust_serial==False:
                drop_serial.append(0)
            elif adjust_serial=='biased':
                init_train=np.sum(Y_train_b)
                misstate_test=np.unique(mini_tbl_test[mini_tbl_test['AAER_DUMMY']==1]['gvkey'])
                Y_train_b[np.isin(mini_tbl_train['gvkey'],misstate_test)]=0
                adj_positive=np.sum(Y_train_b)
                drop_serial.append(init_train-adj_positive)
                
            t01=datetime.now()
            
            clf_lr = Logit(Y_train_b,add_constant(X_train_b)).fit(disp=0)
            predicted_test=clf_lr.predict(add_constant(X_test_b))
            roc_set_ratio.append(roc_auc_score(Y_test_b,predicted_test))
            cutoff_ratio=np.percentile(predicted_test,99)
            labels_ratio=(predicted_test>=cutoff_ratio).astype(int)
            sensitivity_ratio.append(np.sum(np.logical_and(labels_ratio==1, \
                                                         Y_test_b==1))/np.sum(Y_test_b))
            specificity_ratio.append(np.sum(np.logical_and(labels_ratio==0, \
                                                         Y_test_b==0))/np.sum(Y_test_b==0))
            precision_ratio.append(np.sum(np.logical_and(labels_ratio==1, \
                                                         Y_test_b==1))/np.sum(labels_ratio))
            ndcg_ratio.append(ndcg_k(Y_test_b.to_numpy(),predicted_test.to_numpy(),99))
            
            try:
                clf_lr_raw = Logit(Y_train_b,add_constant(X_train_raw_b)).fit(disp=0)
                
                predicted_test_raw=clf_lr_raw.predict(add_constant(X_test_raw_b))
                predicted_test_raw=predicted_test_raw.to_numpy()
            except:
                predicted_test_raw=np.zeros_like(Y_test_b)
            cutoff_raw=np.percentile(predicted_test_raw,99)
            labels_raw=(predicted_test_raw>=cutoff_raw).astype(int)
            sensitivity_raw.append(np.sum(np.logical_and(labels_raw==1, \
                                                         Y_test_b==1))/np.sum(Y_test_b))
            specificity_raw.append(np.sum(np.logical_and(labels_raw==0, \
                                                         Y_test_b==0))/np.sum(Y_test_b==0))
            precision_raw.append(np.sum(np.logical_and(labels_raw==1, \
                                                         Y_test_b==1))/np.sum(labels_raw))
            roc_set_raw.append(roc_auc_score(Y_test_b,predicted_test_raw))
            ndcg_raw.append(ndcg_k(Y_test_b.to_numpy(),predicted_test_raw,99))
         
        
        t1=datetime.now()
        dt=t1-t0
        
        lbls=['mean','std','median','min','max']
        ratio_roc=[np.mean(roc_set_ratio),np.std(roc_set_ratio),np.median(roc_set_ratio),np.min(roc_set_ratio),np.max(roc_set_ratio)]
        raw_roc=[np.mean(roc_set_raw),np.std(roc_set_raw),np.median(roc_set_raw),np.min(roc_set_raw),np.max(roc_set_raw)]
        
        train_positive=[np.mean(count_positive_train),np.std(count_positive_train),
                    np.median(count_positive_train),np.min(count_positive_train),
                    np.max(count_positive_train)]
        test_positive=[np.mean(count_positive_test),np.std(count_positive_test),
                    np.median(count_positive_test),np.min(count_positive_test),
                    np.max(count_positive_test)]
        
        dropped_positive=[np.mean(drop_serial),np.std(drop_serial),
                    np.median(drop_serial),np.min(drop_serial),
                    np.max(drop_serial)]
        
        ratio_sens=[np.mean(sensitivity_ratio),np.std(sensitivity_ratio),
                    np.median(sensitivity_ratio),np.min(sensitivity_ratio),
                    np.max(sensitivity_ratio)]
        raw_sens=[np.mean(sensitivity_raw),np.std(sensitivity_raw),
                    np.median(sensitivity_raw),np.min(sensitivity_raw),
                    np.max(sensitivity_raw)]
        
        ratio_spec=[np.mean(specificity_ratio),np.std(specificity_ratio),
                    np.median(specificity_ratio),np.min(specificity_ratio),
                    np.max(specificity_ratio)]
        raw_spec=[np.mean(specificity_raw),np.std(specificity_raw),
                    np.median(specificity_raw),np.min(specificity_raw),
                    np.max(specificity_raw)]
        
        ratio_prec=[np.mean(precision_ratio),np.std(precision_ratio),
                    np.median(precision_ratio),np.min(precision_ratio),
                    np.max(precision_ratio)]
        raw_prec=[np.mean(precision_raw),np.std(precision_raw),
                    np.median(precision_raw),np.min(precision_raw),
                    np.max(precision_raw)]
        ratio_ndgc=[np.mean(ndcg_ratio),np.std(ndcg_ratio),
                    np.median(ndcg_ratio),np.min(ndcg_ratio),
                    np.max(ndcg_ratio)]
        raw_ndgc=[np.mean(ndcg_raw),np.std(ndcg_raw),
                    np.median(ndcg_raw),np.min(ndcg_raw),
                    np.max(ndcg_raw)]
        result_tbl=pd.DataFrame(data=(lbls,train_positive,test_positive,dropped_positive,
                                      ratio_roc,raw_roc,ratio_sens,raw_sens,
                                      ratio_spec,raw_spec,
                                      ratio_prec,raw_prec,
                                      ratio_ndgc,raw_ndgc)).transpose()
        result_tbl.columns=['measure','training positive','testing positive','dropped positive',
                            'ratio roc','raw roc',
                            'ratio sensitivity @1','raw sensitivity @1',
                            'ratio specificity @1','raw specificity @1',
                            'ratio precision @1','raw precision @1',
                            'ratio NDGC @1','raw NDGC @1']
        result_tbl=result_tbl.set_index('measure')
        result_tbl=result_tbl.transpose()
        
        lbl_perf_tbl='MC_results'+',serial='+str(adjust_serial)+',B='+str(B)+'.csv'
                        
        if write==True:
            result_tbl.to_csv(lbl_perf_tbl,index=True)
        
        t001=datetime.now()
        dt00=t001-t000
        print('MC analysis is completed after '+str(dt00.total_seconds())+' seconds')
        
        ## End of Monte Carlo simulation method
    
    
    def analyse_ratio(self,C_FN=30,C_FP=1):
        """
        This code uses 11 financial ratios to predict the likelihood of fraud in a financial statement.
        
        Parameters:
            – C_FN: Cost of a False Negative for ECM
            – C_FP: Cost of a False Positive for ECM

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
            to 60 mins (you can skip this step) main analysis up to 15 mins. 
            These figures are estimates based on a MacBook Pro 2021.
            
        """

        from sklearn.linear_model import LogisticRegression
        from sklearn.linear_model import SGDClassifier
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import AdaBoostClassifier
        from imblearn.ensemble import RUSBoostClassifier
        from sklearn.model_selection import GridSearchCV,train_test_split
        from sklearn.metrics import roc_auc_score
        from sklearn.tree import DecisionTreeClassifier
        from extra_codes import ndcg_k,relogit
        from statsmodels.discrete.discrete_model import Logit
        from statsmodels.tools import add_constant
        
        t0=datetime.now()
        # setting the parameters
        IS_period=self.ip
        k_fold=self.cv_k
        OOS_period=self.op # 1 year ahead prediction
        OOS_gap=self.og # Gap between training and testing period
        start_OOS_year=self.ts[0]
        end_OOS_year=self.ts[-1]
        sample_start=self.ss
        adjust_serial=self.a_s
        cv_type=self.cv_t
        cross_val=self.cv
        temp_year=self.cv_t_y
        case_window=self.sa
        fraud_df=self.df.copy(deep=True)
        write=self.w

        reduced_tbl_1=fraud_df.iloc[:,[0,1,3,7,8]]
        reduced_tbl_2=fraud_df.iloc[:,-14:-3]
        reduced_tblset=[reduced_tbl_1,reduced_tbl_2]
        reduced_tbl=pd.concat(reduced_tblset,axis=1)
        reduced_tbl=reduced_tbl[reduced_tbl.fyear>=sample_start]
        reduced_tbl=reduced_tbl[reduced_tbl.fyear<=end_OOS_year]

        # Setting the cross-validation setting

        tbl_year_IS_CV=reduced_tbl.loc[np.logical_and(reduced_tbl.fyear<start_OOS_year,\
                                                   reduced_tbl.fyear>=start_OOS_year-IS_period)]
        tbl_year_IS_CV=tbl_year_IS_CV.reset_index(drop=True)
        misstate_firms=np.unique(tbl_year_IS_CV.gvkey[tbl_year_IS_CV.AAER_DUMMY==1])

        X_CV=tbl_year_IS_CV.iloc[:,-11:]

        mean_vals=np.mean(X_CV)
        std_vals=np.std(X_CV)
        X_CV=(X_CV-mean_vals)/std_vals

        Y_CV=tbl_year_IS_CV.AAER_DUMMY

        P_f=np.sum(Y_CV==1)/len(Y_CV)
        P_nf=1-P_f

        print('prior probablity of fraud between '+str(sample_start)+'-'+
              str(start_OOS_year-1)+' is '+str(np.round(P_f*100,2))+'%')

        # redo cross-validation if you wish
        if cv_type=='kfold':
            if cross_val==True: 
                # optimize RUSBoost grid
                print('Grid search hyperparameter optimisation started for RUSBoost')
                t1=datetime.now()
                param_grid_rusboost={'n_estimators':[10,20,50,100,200,500,1000],
                                     'learning_rate':[1e-5,1e-4,1e-3,1e-2,.1,1]}
                base_tree=DecisionTreeClassifier(min_samples_leaf=5)
                bao_RUSboost=RUSBoostClassifier(base_estimator=base_tree,\
                                 sampling_strategy=1,random_state=0)
                clf_rus = GridSearchCV(bao_RUSboost, param_grid_rusboost,scoring='roc_auc',\
                                   n_jobs=-1,cv=k_fold,refit=False)
                clf_rus.fit(X_CV, Y_CV)
                opt_params_rus=clf_rus.best_params_
                n_opt_rus=opt_params_rus['n_estimators']
                r_opt_rus=opt_params_rus['learning_rate']
                score_rus=clf_rus.best_score_
                    
                t2=datetime.now()
                dt=t2-t1
                print('RUSBoost CV finished after '+str(dt.total_seconds())+' sec')
                print('RUSBoost: The optimal number of estimators is '+str(n_opt_rus))
                
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
                
                
                
                print('Computing CV ROC for LR ...')
                t1=datetime.now()
                score_lr=[]            
                for m in range(0,k_fold):
                    train_sample,test_sample=train_test_split(Y_CV,test_size=1/
                                                              k_fold,shuffle=False,random_state=m)
                    X_train=X_CV.iloc[train_sample.index]
                    X_train=add_constant(X_train)
                    Y_train=train_sample
                    X_test=X_CV.iloc[test_sample.index]
                    X_test=add_constant(X_test)
                    Y_test=test_sample
                    
                    logit_model=Logit(Y_train,X_train)
                    logit_model=logit_model.fit(disp=0)
                    pred_LR_CV=logit_model.predict(X_test)
                    score_lr.append(roc_auc_score(Y_test,pred_LR_CV))
                
                    
                score_lr=np.mean(score_lr)
                
                t2=datetime.now()
                dt=t2-t1
                print('LR CV finished after '+str(dt.total_seconds())+' sec')
                
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
                
                param_grid_ada={'n_estimators':[10,20,50,100,200,500],\
                                'learning_rate':[.1,5e-1,9e-1,1]}
                    
                base_lr=LogisticRegression(random_state=0,solver='newton-cg')
                base_mdl_ada=AdaBoostClassifier(base_estimator=base_lr,random_state=0)
                
                clf_ada = GridSearchCV(base_mdl_ada, param_grid_ada,scoring='roc_auc',\
                                   n_jobs=-1,cv=k_fold,refit=False)
                clf_ada.fit(X_CV, Y_CV)
                score_ada=clf_ada.best_score_
                if score_ada>=best_perf_ada:
                    best_perf_ada=score_ada
                    opt_params_ada=clf_ada.best_params_
                
                t2=datetime.now()
                dt=t2-t1
                print('LogitBoost CV finished after '+str(dt.total_seconds())+' sec')
                
                print('LogitBoost: The optimal number of estimators is '+\
                      str(opt_params_ada['n_estimators'])+', and learning rate '+\
                          str(opt_params_ada['learning_rate']))
                
                # optimise MLP classifier
                print('Grid search hyperparameter optimisation started for MLP')
                t1=datetime.now()
                param_grid_mlp={'hidden_layer_sizes':[1,2,5,10],'solver':['sgd','adam'],\
                                'activation':['identity','logistic']}
                base_mdl_mlp=MLPClassifier(random_state=0,validation_fraction=.2)
                
                clf_mlp = GridSearchCV(base_mdl_mlp, param_grid_mlp,scoring='roc_auc',\
                                   n_jobs=-1,cv=k_fold,refit=False)
                clf_mlp.fit(X_CV, Y_CV)
                opt_params_mlp=clf_mlp.best_params_
                score_mlp=clf_mlp.best_score_
                t2=datetime.now()
                dt=t2-t1
                print('MLP CV finished after '+str(dt.total_seconds())+' sec')
                print('MLP: The optimal number of hidden layer is '+\
                      str(opt_params_mlp['hidden_layer_sizes'])+', activation function '+\
                                  opt_params_mlp['activation']+', and solver '+\
                                      opt_params_mlp['solver'])
                
                print('Hyperparameter optimisation finished successfully.\nStarting the main analysis now...')
            else:
                
                n_opt_rus=1000
                r_opt_rus=1e-4
                score_rus=0.6953935928499526
                
                opt_params_svm={'class_weight': {0: 0.01, 1: 1}, 'kernel': 'linear'}
                C_opt=opt_params_svm['class_weight'][0]
                kernel_opt=opt_params_svm['kernel']
                score_svm=0.701939025416111
                
                score_lr=0.7056438104977343
                
                opt_params_sgd={'class_weight': {0: 5e-3, 1: 1}, 'loss': 'log', 'penalty': 'l2'}
                score_sgd=0.7026775920776185
    
                opt_params_ada={'learning_rate': 0.9, 'n_estimators': 20}
                score_ada=0.700229450411913
                
                
                opt_params_mlp={'activation': 'logistic', 'hidden_layer_sizes': 5, 'solver': 'adam'}
                score_mlp=0.706333862286029
                
                
                print('CV skipped ... using defaults for ratio case')
                
        elif cv_type=='temp':
            if cross_val==True: 
                # optimize RUSBoost grid
                cutoff_temporal=2001-temp_year
                X_CV_train=X_CV[tbl_year_IS_CV['fyear']<cutoff_temporal]
                Y_CV_train=Y_CV[tbl_year_IS_CV['fyear']<cutoff_temporal]
                
                X_CV_test=X_CV[tbl_year_IS_CV['fyear']>=cutoff_temporal]
                Y_CV_test=Y_CV[tbl_year_IS_CV['fyear']>=cutoff_temporal]
                
                
                
                print('Grid search hyperparameter optimisation started for RUSBoost')
                t1=datetime.now()
                param_grid_rusboost={'n_estimators':[10,20,50,100,200,500,1000,2000],
                                     'learning_rate':[1e-5,1e-4,1e-3,1e-2,.1,1]}
                
                temp_rusboost={'n_estimators':[],'learning_rate':[],'score':[]}
                
                
                for n in param_grid_rusboost['n_estimators']:
                    for r in param_grid_rusboost['learning_rate']:
                        temp_rusboost['n_estimators'].append(n)
                        temp_rusboost['learning_rate'].append(r)
                        base_tree=DecisionTreeClassifier(min_samples_leaf=5)
                        bao_RUSboost=RUSBoostClassifier(base_estimator=base_tree,\
                                         sampling_strategy=1,n_estimators=n,
                                         learning_rate=r,random_state=0).fit(
                                             X_CV_train,Y_CV_train)
                        predicted_test_RUS=bao_RUSboost.predict_proba(X_CV_test)[:,-1]
                        
                        temp_rusboost['score'].append(roc_auc_score(Y_CV_test,predicted_test_RUS))
                        
                
                
                idx_opt_rus=temp_rusboost['score'].index(np.max(temp_rusboost['score']))
                n_opt_rus=temp_rusboost['n_estimators'][idx_opt_rus]
                r_opt_rus=temp_rusboost['learning_rate'][idx_opt_rus]
                score_rus=temp_rusboost['score'][idx_opt_rus]
                    
                t2=datetime.now()
                dt=t2-t1
                print('RUSBoost Temporal validation finished after '+str(dt.total_seconds())+' sec')
                print('RUSBoost: The optimal number of estimators is '+str(n_opt_rus))
                
                # optimize SVM grid
                
                print('Grid search hyperparameter optimisation started for SVM')
                t1=datetime.now()
                param_grid_svm={'kernel':['linear','rbf','poly'],'class_weight':[\
                                                {0:2e-3,1:1},{0:5e-3,1:1},
                                                {0:1e-2,1:1},{0:2e-2,1:1},{0:5e-2,1:1},\
                                                {0:1e-1,1:1},{0:2e-1,1:1},{0:5e-1,1:1},{0:1e0,1:1}]}
                
                    
                temp_svm={'kernel':[],'class_weight':[],'score':[]}
                
                
                for k in param_grid_svm['kernel']:
                    for w in param_grid_svm['class_weight']:
                        temp_svm['kernel'].append(k)
                        temp_svm['class_weight'].append(w)
                        
                        base_mdl_svm=SVC(shrinking=False,\
                                            probability=False,
                                            kernel=k,class_weight=w,\
                                            random_state=0,max_iter=-1,\
                                                tol=X_CV.shape[-1]*1e-3).fit(X_CV_train,Y_CV_train)
                        predicted_test_svc=base_mdl_svm.decision_function(X_CV_test)
                        
                        predicted_test_svc=np.exp(predicted_test_svc)/(1+np.exp(predicted_test_svc))
                        
                        temp_svm['score'].append(roc_auc_score(Y_CV_test,predicted_test_svc))
                
                
                idx_opt_svm=temp_svm['score'].index(np.max(temp_svm['score']))
                
                C_opt=temp_svm['class_weight'][idx_opt_svm][0]
                kernel_opt=temp_svm['kernel'][idx_opt_svm]
                score_svm=temp_svm['score'][idx_opt_svm]
                opt_params_svm={'class_weight':temp_svm['class_weight'][idx_opt_svm],
                                'kernel':kernel_opt}
                print(opt_params_svm)
                t2=datetime.now()
                dt=t2-t1
                print('SVM Temporal validation finished after '+str(dt.total_seconds())+' sec')
                print('SVM: The optimal C+/C- ratio is '+str(1/C_opt))
                
                
                print('Computing Temporal validation ROC for LR ...')
                t1=datetime.now()
                
                
                logit_model=Logit(Y_CV_train,X_CV_train).fit(disp=0)
                pred_LR_CV=logit_model.predict(X_CV_test)
                score_lr=(roc_auc_score(Y_CV_test,pred_LR_CV))
                    
                
                t2=datetime.now()
                dt=t2-t1
                print('LR Temporal validation finished after '+str(dt.total_seconds())+' sec')
                
                # optimise SGD
                print('Grid search hyperparameter optimisation started for SGD')
                t1=datetime.now()
                
                param_grid_sgd={'penalty':['l1','l2'],'loss':['log','modified_huber'],\
                                'class_weight':[{0:2e-3,1:1},{0:5e-3,1:1},{0:1e-2,1:1},\
                                                {0:2e-2,1:1},{0:5e-2,1:1},{0:1e-1,1:1},\
                                                    {0:2e-1,1:1},{0:5e-1,1:1},{0:1e0,1:1}]}
                
                temp_sgd={'penalty':[],'loss':[],'class_weight':[],'score':[]}
                
                
                for p in param_grid_sgd['penalty']:
                    for l in param_grid_sgd['loss']:
                        for w in param_grid_sgd['class_weight']:
                            temp_sgd['penalty'].append(p)
                            temp_sgd['loss'].append(l)
                            temp_sgd['class_weight'].append(w)
                        
                            base_mdl_sgd=SGDClassifier(random_state=0,
                                                       validation_fraction=.2,
                                                       shuffle=False,
                                                       penalty=p,
                                                       loss=l,
                                                       class_weight=w).fit(
                                                           X_CV_train,Y_CV_train)
                            predicted_test_sgd=base_mdl_sgd.predict_proba(X_CV_test)[:,-1]
                            temp_sgd['score'].append(roc_auc_score(Y_CV_test,predicted_test_sgd))
                        
                
                idx_opt_sgd=temp_sgd['score'].index(np.max(temp_sgd['score']))
                
                score_sgd=temp_sgd['score'][idx_opt_sgd]
                opt_params_sgd={'class_weight':temp_sgd['class_weight'][idx_opt_sgd],
                                'loss':temp_sgd['loss'][idx_opt_sgd],
                                'penalty':temp_sgd['penalty'][idx_opt_sgd]}
                
                t2=datetime.now()
                dt=t2-t1
                print('SGD Temporal validation finished after '+str(dt.total_seconds())+' sec')
                print('SGD: The optimal C+/C- ratio is '+str(1/opt_params_sgd['class_weight'][0])+\
                      ', loss function is '+opt_params_sgd['loss']+', and penalty '+\
                          opt_params_sgd['penalty'])
                
                # optimise LogitBoost
                print('Grid search hyperparameter optimisation started for LogitBoost')
                t1=datetime.now()
                
                
                param_grid_ada={'n_estimators':[10,20,50,100,200,500],\
                                'learning_rate':[.1,5e-1,9e-1,1]}
                
                temp_ada={'n_estimators':[],'learning_rate':[],'score':[]}
                
                for n in param_grid_ada['n_estimators']:
                    for r in param_grid_ada['learning_rate']:
                            temp_ada['n_estimators'].append(n)
                            temp_ada['learning_rate'].append(r)
                            
                            base_lr=LogisticRegression(random_state=0,solver='newton-cg')
                            base_mdl_ada=AdaBoostClassifier(base_estimator=base_lr,
                                                            learning_rate=r,
                                                            n_estimators=n,
                                                            random_state=0).fit(
                                                                X_CV_train,
                                                                Y_CV_train)
                            predicted_ada_test=base_mdl_ada.predict_proba(X_CV_test)[:,-1]
                            temp_ada['score'].append(roc_auc_score(Y_CV_test,predicted_ada_test))
                            
                            
                    
                idx_opt_ada=temp_ada['score'].index(np.max(temp_ada['score']))
                
                score_ada=temp_ada['score'][idx_opt_ada]
                opt_params_ada={'n_estimators':temp_ada['n_estimators'][idx_opt_ada],
                                'learning_rate':temp_ada['learning_rate'][idx_opt_ada]}
                    
                
                t2=datetime.now()
                dt=t2-t1
                print('LogitBoost Temporal validation finished after '+str(dt.total_seconds())+' sec')
                
                print('LogitBoost: The optimal number of estimators is '+\
                      str(opt_params_ada['n_estimators'])+', and learning rate '+\
                          str(opt_params_ada['learning_rate']))
                
                # optimise MLP classifier
                print('Grid search hyperparameter optimisation started for MLP')
                t1=datetime.now()
                param_grid_mlp={'hidden_layer_sizes':[1,2,5,10],'solver':['sgd','adam'],\
                                'activation':['identity','logistic']}
                    
                temp_mlp={'hidden_layer_sizes':[],'solver':[],'activation':[],'score':[]}
                
                for h in param_grid_mlp['hidden_layer_sizes']:
                    for s in param_grid_mlp['solver']:
                        for a in param_grid_mlp['activation']:
                            temp_mlp['hidden_layer_sizes'].append(h)
                            temp_mlp['solver'].append(s)
                            temp_mlp['activation'].append(a)
                        
                            base_mdl_mlp=MLPClassifier(random_state=0,
                                                       validation_fraction=.2,
                                                       hidden_layer_sizes=h,
                                                       solver=s,
                                                       activation=a).fit(X_CV_train,
                                                                         Y_CV_train)
                            predicted_mlp_test=base_mdl_mlp.predict_proba(X_CV_test)[:,-1]
                            temp_mlp['score'].append(roc_auc_score(Y_CV_test,predicted_mlp_test))
                
                
                idx_opt=temp_mlp['score'].index(np.max(temp_mlp['score']))
                
                score_mlp=temp_mlp['score'][idx_opt]
                opt_params_mlp={'hidden_layer_sizes':temp_mlp['hidden_layer_sizes'][idx_opt],
                                'solver':temp_mlp['solver'][idx_opt],
                                'activation':temp_mlp['activation'][idx_opt]}
                
                
                
                t2=datetime.now()
                dt=t2-t1
                print('MLP Temporal validation finished after '+str(dt.total_seconds())+' sec')
                print('MLP: The optimal number of hidden layer is '+\
                      str(opt_params_mlp['hidden_layer_sizes'])+', activation function '+\
                                  opt_params_mlp['activation']+', and solver '+\
                                      opt_params_mlp['solver'])
                
                print('Hyperparameter optimisation finished successfully.\nStarting the main analysis now...')
            else:
                n_opt_rus=500
                r_opt_rus=1e-1
                score_rus=0.7002526951115313
                
                # n_opt_rus=500
                # r_opt_rus=1
                # score_rus=0.6615684777537431
                
                opt_params_svm={'class_weight': {0: 0.01, 1: 1}, 'kernel': 'linear'}
                C_opt=opt_params_svm['class_weight'][0]
                kernel_opt=opt_params_svm['kernel']
                score_svm=0.6701583434835566
                
                
                score_lr=0.5022597124002399
                # score_lr=0.47810153963608604
                
                opt_params_sgd={'class_weight': {0: 0.5, 1: 1}, 'loss': 'log', 'penalty': 'l1'}
                score_sgd=0.6803715890704819
                # opt_params_sgd={'class_weight': {0: 0.5, 1: 1}, 'loss': 'log', 'penalty': 'l2'}
                # score_sgd=0.6872969419349366
    
                opt_params_ada={'n_estimators': 200, 'learning_rate': 0.9}
                score_ada=0.6641663788245132
                
                
                opt_params_mlp={'hidden_layer_sizes': 2, 'solver': 'adam', 'activation': 'identity'}
                score_mlp=0.6660970421946298
                
                # opt_params_mlp={'hidden_layer_sizes': 5, 'solver': 'adam', 'activation': 'logistic'}
                # score_mlp=0.671069262416762
                
                print('Temporal validation skipped ... using defaults for ratio case')
                
            

        range_oos=range(start_OOS_year,end_OOS_year+1,OOS_period)

        roc_rus=np.zeros(len(range_oos))
        sensitivity_OOS_rus1=np.zeros(len(range_oos))
        specificity_OOS_rus1=np.zeros(len(range_oos))
        precision_rus1=np.zeros(len(range_oos))
        ndcg_rus1=np.zeros(len(range_oos))
        ecm_rus1=np.zeros(len(range_oos))


        roc_svm=np.zeros(len(range_oos))
        sensitivity_OOS_svm1=np.zeros(len(range_oos))
        specificity_OOS_svm1=np.zeros(len(range_oos))
        precision_svm1=np.zeros(len(range_oos))
        ndcg_svm1=np.zeros(len(range_oos))
        ecm_svm1=np.zeros(len(range_oos))

        roc_lr=np.zeros(len(range_oos))
        sensitivity_OOS_lr1=np.zeros(len(range_oos))
        specificity_OOS_lr1=np.zeros(len(range_oos))
        precision_lr1=np.zeros(len(range_oos))
        ndcg_lr1=np.zeros(len(range_oos))
        ecm_lr1=np.zeros(len(range_oos))
        

        roc_sgd=np.zeros(len(range_oos))
        sensitivity_OOS_sgd1=np.zeros(len(range_oos))
        specificity_OOS_sgd1=np.zeros(len(range_oos))
        precision_sgd1=np.zeros(len(range_oos))
        ndcg_sgd1=np.zeros(len(range_oos))
        ecm_sgd1=np.zeros(len(range_oos))

        roc_ada=np.zeros(len(range_oos))
        sensitivity_OOS_ada1=np.zeros(len(range_oos))
        specificity_OOS_ada1=np.zeros(len(range_oos))
        precision_ada1=np.zeros(len(range_oos))
        ndcg_ada1=np.zeros(len(range_oos))
        ecm_ada1=np.zeros(len(range_oos))


        roc_mlp=np.zeros(len(range_oos))
        sensitivity_OOS_mlp1=np.zeros(len(range_oos))
        specificity_OOS_mlp1=np.zeros(len(range_oos))
        precision_mlp1=np.zeros(len(range_oos))
        ndcg_mlp1=np.zeros(len(range_oos))
        ecm_mlp1=np.zeros(len(range_oos))


        roc_fused=np.zeros(len(range_oos))
        sensitivity_OOS_fused1=np.zeros(len(range_oos))
        specificity_OOS_fused1=np.zeros(len(range_oos))
        precision_fused1=np.zeros(len(range_oos))
        ndcg_fused1=np.zeros(len(range_oos))
        ecm_fused1=np.zeros(len(range_oos))


        m=0
        for yr in range_oos:
            t1=datetime.now()
            if case_window=='expanding':
                year_start_IS=sample_start
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
            bao_RUSboost=RUSBoostClassifier(base_estimator=base_tree,n_estimators=n_opt_rus,\
                             learning_rate=r_opt_rus,sampling_strategy=1,random_state=0)
            clf_rusboost=bao_RUSboost.fit(X, Y)
            probs_oos_fraud_rus=clf_rusboost.predict_proba(X_OOS)[:,-1]
            roc_rus[m]=roc_auc_score(Y_OOS,probs_oos_fraud_rus)
                        
            
            
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
            
            roc_svm[m]=roc_auc_score(Y_OOS,probs_oos_fraud_svm)
            
            
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
            X_lr=add_constant(X)
            X_OOS_lr=add_constant(X_OOS)
            clf_lr = Logit(Y,X_lr)
            clf_lr=clf_lr.fit(disp=0)
            probs_oos_fraud_lr=clf_lr.predict(X_OOS_lr)

            roc_lr[m]=roc_auc_score(Y_OOS,probs_oos_fraud_lr)
            
            
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
            
            roc_sgd[m]=roc_auc_score(Y_OOS,probs_oos_fraud_sgd)
            
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
            base_lr=LogisticRegression(random_state=0,solver='newton-cg')
            
            
            clf_ada=AdaBoostClassifier(n_estimators=opt_params_ada['n_estimators'],\
                                       learning_rate=opt_params_ada['learning_rate'],\
                                           base_estimator=base_lr,random_state=0)
            clf_ada=clf_ada.fit(X,Y)
            probs_oos_fraud_ada=clf_ada.predict_proba(X_OOS)[:,-1]
            
            
            labels_ada=clf_ada.predict(X_OOS)
            
            roc_ada[m]=roc_auc_score(Y_OOS,probs_oos_fraud_ada)
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
            clf_mlp=MLPClassifier(hidden_layer_sizes=opt_params_mlp['hidden_layer_sizes'], \
                                  activation=opt_params_mlp['activation'],solver=opt_params_mlp['solver'],\
                                               random_state=0,validation_fraction=.1)
            clf_mlp=clf_mlp.fit(X,Y)
            probs_oos_fraud_mlp=clf_mlp.predict_proba(X_OOS)[:,-1]
                        
            roc_mlp[m]=roc_auc_score(Y_OOS,probs_oos_fraud_mlp)
            
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
            
            probs_oos_fraud_svm=(1+np.exp(-1*probs_oos_fraud_svm))**-1
                
            probs_oos_fraud_lr=(1+np.exp(-1*probs_oos_fraud_lr))**-1
                
            probs_oos_fraud_sgd=(1+np.exp(-1*probs_oos_fraud_sgd))**-1
            
            probs_oos_fraud_ada=(1+np.exp(-1*probs_oos_fraud_ada))**-1
                
            probs_oos_fraud_mlp=(1+np.exp(-1*probs_oos_fraud_mlp))**-1
            
                
            clf_fused=np.dot(np.array([probs_oos_fraud_svm,\
                                  probs_oos_fraud_lr,probs_oos_fraud_sgd,probs_oos_fraud_ada,\
                                      probs_oos_fraud_mlp]).T,weight_ser)
            
            probs_oos_fraud_fused=clf_fused
                        
            roc_fused[m]=roc_auc_score(Y_OOS,probs_oos_fraud_fused)
            
            
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

        f1_score_rus1=2*(precision_rus1*sensitivity_OOS_rus1)/\
            (precision_rus1+sensitivity_OOS_rus1+1e-8)
        
        f1_score_svm1=2*(precision_svm1*sensitivity_OOS_svm1)/\
            (precision_svm1+sensitivity_OOS_svm1+1e-8)
        
        f1_score_lr1=2*(precision_lr1*sensitivity_OOS_lr1)/\
            (precision_lr1+sensitivity_OOS_lr1+1e-8)
        
        f1_score_sgd1=2*(precision_sgd1*sensitivity_OOS_sgd1)/\
            (precision_sgd1+sensitivity_OOS_sgd1+1e-8)
        
        f1_score_ada1=2*(precision_ada1*sensitivity_OOS_ada1)/\
            (precision_ada1+sensitivity_OOS_ada1+1e-8)
        
        f1_score_mlp1=2*(precision_mlp1*sensitivity_OOS_mlp1)/\
            (precision_mlp1+sensitivity_OOS_mlp1+1e-8)
        
        f1_score_fused1=2*(precision_fused1*sensitivity_OOS_fused1)/\
            (precision_fused1+sensitivity_OOS_fused1+1e-8)
        
        # create performance table now
        perf_tbl_general=pd.DataFrame()
        perf_tbl_general['models']=['RUSBoost','SVM','LR','SGD','LogitBoost','MLP','FUSED']
        perf_tbl_general['Roc']=[str(np.round(
            np.mean(roc_rus)*100,2))+'% ('+\
            str(np.round(np.std(roc_rus)*100,2))+'%)',str(np.round(
                np.mean(roc_svm)*100,2))+'% ('+\
                str(np.round(np.std(roc_svm)*100,2))+'%)',str(np.round(
                    np.mean(roc_lr)*100,2))+'% ('+\
                    str(np.round(np.std(roc_lr)*100,2))+'%)',str(np.round(
                        np.mean(roc_sgd)*100,2))+'% ('+\
                        str(np.round(np.std(roc_sgd)*100,2))+'%)',str(np.round(
                            np.mean(roc_ada)*100,2))+'% ('+\
                            str(np.round(np.std(roc_ada)*100,2))+'%)',str(np.round(
                                np.mean(roc_mlp)*100,2))+'% ('+\
                                str(np.round(np.std(roc_mlp)*100,2))+'%)',
                                str(np.round(
                                    np.mean(roc_fused)*100,2))+'% ('+\
                                    str(np.round(np.std(roc_fused)*100,2))+'%)']
        
        

                                                    
        perf_tbl_general['Sensitivity @ 1 Prc']=[str(np.round(
            np.mean(sensitivity_OOS_rus1)*100,2))+'% ('+\
            str(np.round(np.std(sensitivity_OOS_rus1)*100,2))+'%)',str(np.round(
                np.mean(sensitivity_OOS_svm1)*100,2))+'% ('+\
                str(np.round(np.std(sensitivity_OOS_svm1)*100,2))+'%)',str(np.round(
                    np.mean(sensitivity_OOS_lr1)*100,2))+'% ('+\
                    str(np.round(np.std(sensitivity_OOS_lr1)*100,2))+'%)',str(np.round(
                        np.mean(sensitivity_OOS_sgd1)*100,2))+'% ('+\
                        str(np.round(np.std(sensitivity_OOS_sgd1)*100,2))+'%)',str(np.round(
                            np.mean(sensitivity_OOS_ada1)*100,2))+'% ('+\
                            str(np.round(np.std(sensitivity_OOS_ada1)*100,2))+'%)',str(np.round(
                                np.mean(sensitivity_OOS_mlp1)*100,2))+'% ('+\
                                str(np.round(np.std(sensitivity_OOS_mlp1)*100,2))+'%)',
                                str(np.round(
                                    np.mean(sensitivity_OOS_fused1)*100,2))+'% ('+\
                                    str(np.round(np.std(sensitivity_OOS_fused1)*100,2))+'%)']
        
        

        perf_tbl_general['Specificity @ 1 Prc']=[str(np.round(
            np.mean(specificity_OOS_rus1)*100,2))+'% ('+\
            str(np.round(np.std(specificity_OOS_rus1)*100,2))+'%)',str(np.round(
                np.mean(specificity_OOS_svm1)*100,2))+'% ('+\
                str(np.round(np.std(specificity_OOS_svm1)*100,2))+'%)',str(np.round(
                    np.mean(specificity_OOS_lr1)*100,2))+'% ('+\
                    str(np.round(np.std(specificity_OOS_lr1)*100,2))+'%)',str(np.round(
                        np.mean(specificity_OOS_sgd1)*100,2))+'% ('+\
                        str(np.round(np.std(specificity_OOS_sgd1)*100,2))+'%)',str(np.round(
                            np.mean(specificity_OOS_ada1)*100,2))+'% ('+\
                            str(np.round(np.std(specificity_OOS_ada1)*100,2))+'%)',str(np.round(
                                np.mean(specificity_OOS_mlp1)*100,2))+'% ('+\
                                str(np.round(np.std(specificity_OOS_mlp1)*100,2))+'%)',
                                str(np.round(
                                    np.mean(specificity_OOS_fused1)*100,2))+'% ('+\
                                    str(np.round(np.std(specificity_OOS_fused1)*100,2))+'%)']
        
        
        

        perf_tbl_general['Precision @ 1 Prc']=[str(np.round(
            np.mean(precision_rus1)*100,2))+'% ('+\
            str(np.round(np.std(precision_rus1)*100,2))+'%)',str(np.round(
                np.mean(precision_svm1)*100,2))+'% ('+\
                str(np.round(np.std(precision_svm1)*100,2))+'%)',str(np.round(
                    np.mean(precision_lr1)*100,2))+'% ('+\
                    str(np.round(np.std(precision_lr1)*100,2))+'%)',str(np.round(
                        np.mean(precision_sgd1)*100,2))+'% ('+\
                        str(np.round(np.std(precision_sgd1)*100,2))+'%)',str(np.round(
                            np.mean(precision_ada1)*100,2))+'% ('+\
                            str(np.round(np.std(precision_ada1)*100,2))+'%)',str(np.round(
                                np.mean(precision_mlp1)*100,2))+'% ('+\
                                str(np.round(np.std(precision_mlp1)*100,2))+'%)',
                                str(np.round(
                                    np.mean(precision_fused1)*100,2))+'% ('+\
                                    str(np.round(np.std(precision_fused1)*100,2))+'%)']

        perf_tbl_general['F1 Score @ 1 Prc']=[str(np.round(
            np.mean(f1_score_rus1)*100,2))+'% ('+\
            str(np.round(np.std(f1_score_rus1)*100,2))+'%)',str(np.round(
                np.mean(f1_score_svm1)*100,2))+'% ('+\
                str(np.round(np.std(f1_score_svm1)*100,2))+'%)',str(np.round(
                    np.mean(f1_score_lr1)*100,2))+'% ('+\
                    str(np.round(np.std(f1_score_lr1)*100,2))+'%)',str(np.round(
                        np.mean(f1_score_sgd1)*100,2))+'% ('+\
                        str(np.round(np.std(f1_score_sgd1)*100,2))+'%)',str(np.round(
                            np.mean(f1_score_ada1)*100,2))+'% ('+\
                            str(np.round(np.std(f1_score_ada1)*100,2))+'%)',str(np.round(
                                np.mean(f1_score_mlp1)*100,2))+'% ('+\
                                str(np.round(np.std(f1_score_mlp1)*100,2))+'%)',
                                str(np.round(
                                    np.mean(f1_score_fused1)*100,2))+'% ('+\
                                    str(np.round(np.std(f1_score_fused1)*100,2))+'%)']
            
        
        perf_tbl_general['NDCG @ 1 Prc']=[str(np.round(
            np.mean(ndcg_rus1)*100,2))+'% ('+\
            str(np.round(np.std(ndcg_rus1)*100,2))+'%)',str(np.round(
                np.mean(ndcg_svm1)*100,2))+'% ('+\
                str(np.round(np.std(ndcg_svm1)*100,2))+'%)',str(np.round(
                    np.mean(ndcg_lr1)*100,2))+'% ('+\
                    str(np.round(np.std(ndcg_lr1)*100,2))+'%)',str(np.round(
                        np.mean(ndcg_sgd1)*100,2))+'% ('+\
                        str(np.round(np.std(ndcg_sgd1)*100,2))+'%)',str(np.round(
                            np.mean(ndcg_ada1)*100,2))+'% ('+\
                            str(np.round(np.std(ndcg_ada1)*100,2))+'%)',str(np.round(
                                np.mean(ndcg_mlp1)*100,2))+'% ('+\
                                str(np.round(np.std(ndcg_mlp1)*100,2))+'%)',
                                str(np.round(
                                    np.mean(ndcg_fused1)*100,2))+'% ('+\
                                    str(np.round(np.std(ndcg_fused1)*100,2))+'%)']
        
        
        
        
        
        perf_tbl_general['ECM @ 1 Prc']=[str(np.round(
            np.mean(ecm_rus1)*100,2))+'% ('+\
            str(np.round(np.std(ecm_rus1)*100,2))+'%)',str(np.round(
                np.mean(ecm_svm1)*100,2))+'% ('+\
                str(np.round(np.std(ecm_svm1)*100,2))+'%)',str(np.round(
                    np.mean(ecm_lr1)*100,2))+'% ('+\
                    str(np.round(np.std(ecm_lr1)*100,2))+'%)',str(np.round(
                        np.mean(ecm_sgd1)*100,2))+'% ('+\
                        str(np.round(np.std(ecm_sgd1)*100,2))+'%)',str(np.round(
                            np.mean(ecm_ada1)*100,2))+'% ('+\
                            str(np.round(np.std(ecm_ada1)*100,2))+'%)',str(np.round(
                                np.mean(ecm_mlp1)*100,2))+'% ('+\
                                str(np.round(np.std(ecm_mlp1)*100,2))+'%)',
                                str(np.round(
                                    np.mean(ecm_fused1)*100,2))+'% ('+\
                                    str(np.round(np.std(ecm_fused1)*100,2))+'%)']
    
            
        

        if cv_type=='kfold':
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
        else:
            if case_window=='expanding':
                lbl_perf_tbl='perf_tbl_'+str(start_OOS_year)+'_'+str(end_OOS_year)+\
                    '_'+case_window+',OOS='+str(OOS_period)+','+\
                    'temporal'+',serial='+str(adjust_serial)+\
                    ',gap='+str(OOS_gap)+'_11ratios.csv'
            else:
                lbl_perf_tbl='perf_tbl_'+str(start_OOS_year)+'_'+str(end_OOS_year)+\
                    '_IS='+str(IS_period)+',OOS='+str(OOS_period)+','+\
                    'temporal'+',serial='+str(adjust_serial)+\
                    ',gap='+str(OOS_gap)+'_11ratios.csv'

        if write==True:
            perf_tbl_general.to_csv(lbl_perf_tbl,index=False)
        print(perf_tbl_general)
        t_last=datetime.now()
        dt_total=t_last-t0
        print('total run time is '+str(dt_total.total_seconds())+' sec')


        # extract performance for 2003-2008 directly from 2001-2010

        perf_tbl_general=pd.DataFrame()
        perf_tbl_general['models']=['RUSBoost','SVM','LR','SGD','LogitBoost','MLP','FUSED']
        
        
        perf_tbl_general['Roc']=[str(np.round(
            np.mean(roc_rus[2:8])*100,2))+'% ('+\
            str(np.round(np.std(roc_rus[2:8])*100,2))+'%)',str(np.round(
                np.mean(roc_svm[2:8])*100,2))+'% ('+\
                str(np.round(np.std(roc_svm[2:8])*100,2))+'%)',str(np.round(
                    np.mean(roc_lr[2:8])*100,2))+'% ('+\
                    str(np.round(np.std(roc_lr[2:8])*100,2))+'%)',str(np.round(
                        np.mean(roc_sgd[2:8])*100,2))+'% ('+\
                        str(np.round(np.std(roc_sgd[2:8])*100,2))+'%)',str(np.round(
                            np.mean(roc_ada[2:8])*100,2))+'% ('+\
                            str(np.round(np.std(roc_ada[2:8])*100,2))+'%)',str(np.round(
                                np.mean(roc_mlp[2:8])*100,2))+'% ('+\
                                str(np.round(np.std(roc_mlp[2:8])*100,2))+'%)',
                                str(np.round(
                                    np.mean(roc_fused[2:8])*100,2))+'% ('+\
                                    str(np.round(np.std(roc_fused[2:8])*100,2))+'%)']
        
        

                                                    
        perf_tbl_general['Sensitivity @ 1 Prc']=[str(np.round(
            np.mean(sensitivity_OOS_rus1[2:8])*100,2))+'% ('+\
            str(np.round(np.std(sensitivity_OOS_rus1[2:8])*100,2))+'%)',str(np.round(
                np.mean(sensitivity_OOS_svm1[2:8])*100,2))+'% ('+\
                str(np.round(np.std(sensitivity_OOS_svm1[2:8])*100,2))+'%)',str(np.round(
                    np.mean(sensitivity_OOS_lr1[2:8])*100,2))+'% ('+\
                    str(np.round(np.std(sensitivity_OOS_lr1[2:8])*100,2))+'%)',str(np.round(
                        np.mean(sensitivity_OOS_sgd1[2:8])*100,2))+'% ('+\
                        str(np.round(np.std(sensitivity_OOS_sgd1[2:8])*100,2))+'%)',str(np.round(
                            np.mean(sensitivity_OOS_ada1[2:8])*100,2))+'% ('+\
                            str(np.round(np.std(sensitivity_OOS_ada1[2:8])*100,2))+'%)',str(np.round(
                                np.mean(sensitivity_OOS_mlp1[2:8])*100,2))+'% ('+\
                                str(np.round(np.std(sensitivity_OOS_mlp1[2:8])*100,2))+'%)',
                                str(np.round(
                                    np.mean(sensitivity_OOS_fused1[2:8])*100,2))+'% ('+\
                                    str(np.round(np.std(sensitivity_OOS_fused1[2:8])*100,2))+'%)']
        
        

        perf_tbl_general['Specificity @ 1 Prc']=[str(np.round(
            np.mean(specificity_OOS_rus1[2:8])*100,2))+'% ('+\
            str(np.round(np.std(specificity_OOS_rus1[2:8])*100,2))+'%)',str(np.round(
                np.mean(specificity_OOS_svm1[2:8])*100,2))+'% ('+\
                str(np.round(np.std(specificity_OOS_svm1[2:8])*100,2))+'%)',str(np.round(
                    np.mean(specificity_OOS_lr1[2:8])*100,2))+'% ('+\
                    str(np.round(np.std(specificity_OOS_lr1[2:8])*100,2))+'%)',str(np.round(
                        np.mean(specificity_OOS_sgd1[2:8])*100,2))+'% ('+\
                        str(np.round(np.std(specificity_OOS_sgd1[2:8])*100,2))+'%)',str(np.round(
                            np.mean(specificity_OOS_ada1[2:8])*100,2))+'% ('+\
                            str(np.round(np.std(specificity_OOS_ada1[2:8])*100,2))+'%)',str(np.round(
                                np.mean(specificity_OOS_mlp1[2:8])*100,2))+'% ('+\
                                str(np.round(np.std(specificity_OOS_mlp1[2:8])*100,2))+'%)',
                                str(np.round(
                                    np.mean(specificity_OOS_fused1[2:8])*100,2))+'% ('+\
                                    str(np.round(np.std(specificity_OOS_fused1[2:8])*100,2))+'%)']
        
        
        

        perf_tbl_general['Precision @ 1 Prc']=[str(np.round(
            np.mean(precision_rus1[2:8])*100,2))+'% ('+\
            str(np.round(np.std(precision_rus1[2:8])*100,2))+'%)',str(np.round(
                np.mean(precision_svm1[2:8])*100,2))+'% ('+\
                str(np.round(np.std(precision_svm1[2:8])*100,2))+'%)',str(np.round(
                    np.mean(precision_lr1[2:8])*100,2))+'% ('+\
                    str(np.round(np.std(precision_lr1[2:8])*100,2))+'%)',str(np.round(
                        np.mean(precision_sgd1[2:8])*100,2))+'% ('+\
                        str(np.round(np.std(precision_sgd1[2:8])*100,2))+'%)',str(np.round(
                            np.mean(precision_ada1[2:8])*100,2))+'% ('+\
                            str(np.round(np.std(precision_ada1[2:8])*100,2))+'%)',str(np.round(
                                np.mean(precision_mlp1[2:8])*100,2))+'% ('+\
                                str(np.round(np.std(precision_mlp1[2:8])*100,2))+'%)',
                                str(np.round(
                                    np.mean(precision_fused1[2:8])*100,2))+'% ('+\
                                    str(np.round(np.std(precision_fused1[2:8])*100,2))+'%)']

        perf_tbl_general['F1 Score @ 1 Prc']=[str(np.round(
            np.mean(f1_score_rus1[2:8])*100,2))+'% ('+\
            str(np.round(np.std(f1_score_rus1[2:8])*100,2))+'%)',str(np.round(
                np.mean(f1_score_svm1[2:8])*100,2))+'% ('+\
                str(np.round(np.std(f1_score_svm1[2:8])*100,2))+'%)',str(np.round(
                    np.mean(f1_score_lr1[2:8])*100,2))+'% ('+\
                    str(np.round(np.std(f1_score_lr1[2:8])*100,2))+'%)',str(np.round(
                        np.mean(f1_score_sgd1[2:8])*100,2))+'% ('+\
                        str(np.round(np.std(f1_score_sgd1[2:8])*100,2))+'%)',str(np.round(
                            np.mean(f1_score_ada1[2:8])*100,2))+'% ('+\
                            str(np.round(np.std(f1_score_ada1[2:8])*100,2))+'%)',str(np.round(
                                np.mean(f1_score_mlp1[2:8])*100,2))+'% ('+\
                                str(np.round(np.std(f1_score_mlp1[2:8])*100,2))+'%)',
                                str(np.round(
                                    np.mean(f1_score_fused1[2:8])*100,2))+'% ('+\
                                    str(np.round(np.std(f1_score_fused1[2:8])*100,2))+'%)']
            
        
        perf_tbl_general['NDCG @ 1 Prc']=[str(np.round(
            np.mean(ndcg_rus1[2:8])*100,2))+'% ('+\
            str(np.round(np.std(ndcg_rus1[2:8])*100,2))+'%)',str(np.round(
                np.mean(ndcg_svm1[2:8])*100,2))+'% ('+\
                str(np.round(np.std(ndcg_svm1[2:8])*100,2))+'%)',str(np.round(
                    np.mean(ndcg_lr1[2:8])*100,2))+'% ('+\
                    str(np.round(np.std(ndcg_lr1[2:8])*100,2))+'%)',str(np.round(
                        np.mean(ndcg_sgd1[2:8])*100,2))+'% ('+\
                        str(np.round(np.std(ndcg_sgd1[2:8])*100,2))+'%)',str(np.round(
                            np.mean(ndcg_ada1[2:8])*100,2))+'% ('+\
                            str(np.round(np.std(ndcg_ada1[2:8])*100,2))+'%)',str(np.round(
                                np.mean(ndcg_mlp1[2:8])*100,2))+'% ('+\
                                str(np.round(np.std(ndcg_mlp1[2:8])*100,2))+'%)',
                                str(np.round(
                                    np.mean(ndcg_fused1[2:8])*100,2))+'% ('+\
                                    str(np.round(np.std(ndcg_fused1[2:8])*100,2))+'%)']
        
        
        
        
        
        perf_tbl_general['ECM @ 1 Prc']=[str(np.round(
            np.mean(ecm_rus1[2:8])*100,2))+'% ('+\
            str(np.round(np.std(ecm_rus1[2:8])*100,2))+'%)',str(np.round(
                np.mean(ecm_svm1[2:8])*100,2))+'% ('+\
                str(np.round(np.std(ecm_svm1[2:8])*100,2))+'%)',str(np.round(
                    np.mean(ecm_lr1[2:8])*100,2))+'% ('+\
                    str(np.round(np.std(ecm_lr1[2:8])*100,2))+'%)',str(np.round(
                        np.mean(ecm_sgd1[2:8])*100,2))+'% ('+\
                        str(np.round(np.std(ecm_sgd1[2:8])*100,2))+'%)',str(np.round(
                            np.mean(ecm_ada1[2:8])*100,2))+'% ('+\
                            str(np.round(np.std(ecm_ada1[2:8])*100,2))+'%)',str(np.round(
                                np.mean(ecm_mlp1[2:8])*100,2))+'% ('+\
                                str(np.round(np.std(ecm_mlp1[2:8])*100,2))+'%)',
                                str(np.round(
                                    np.mean(ecm_fused1[2:8])*100,2))+'% ('+\
                                    str(np.round(np.std(ecm_fused1[2:8])*100,2))+'%)']

        if cv_type=='kfold':
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
        else:
            if case_window=='expanding':
                lbl_perf_tbl='perf_tbl_'+str(2003)+'_'+str(2008)+\
                    '_'+case_window+',OOS='+str(OOS_period)+','+\
                    'temporal'+',serial='+str(adjust_serial)+\
                    ',gap='+str(OOS_gap)+'_11ratios.csv'
            else:
                lbl_perf_tbl='perf_tbl_'+str(2003)+'_'+str(2008)+\
                    '_IS='+str(IS_period)+',OOS='+str(OOS_period)+','+\
                    'temporal'+',serial='+str(adjust_serial)+\
                    ',gap='+str(OOS_gap)+'_11ratios.csv'
        if write==True:
            perf_tbl_general.to_csv(lbl_perf_tbl,index=False)

        ## End of analysis of 11 ratios procedure 
        
    def analyse_raw(self, C_FN=30,C_FP=1):
        """
        This code replicates the RUSBoost model of Bao et al (2020).
        Skipping cross-validation sets the number of estimators to 1000.

        Parameters:
            – C_FN: Cost of a False Negative for ECM
            – C_FP: Cost of a False Positive for ECM
        
        
        Predictive models:
            – RUSBoost based on Scikit module
        Outputs: 
        Main results are stored in the table variable "perf_tbl_general" written into
        2 csv files: time period 2001-2010 and 2003-2008

        Steps:
            1. Cross-validate to find optimal hyperparameters.
            2. Estimating the performance for each OOS period.

        Warnings: 
            – Running this code can take up to 10 mins when CV is skipped. 
            These figures are estimates based on a MacBook Pro 2021.
            
        """
        
        
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import GridSearchCV
        from datetime import datetime
        from imblearn.ensemble import RUSBoostClassifier
        from sklearn.metrics import roc_auc_score
        from extra_codes import ndcg_k
        
        t0=datetime.now()

        ## setting the parameters
        
        # IS_period=self.ip since the Bao approach is an expanding one
        k_fold=self.cv_k
        OOS_period=self.op # 1 year ahead prediction
        OOS_gap=self.og # Gap between training and testing period
        start_OOS_year=self.ts[0]
        end_OOS_year=self.ts[-1]
        sample_start=self.ss
        adjust_serial=self.a_s
        cv_type=self.cv_t
        cross_val=self.cv
        temp_year=self.cv_t_y
        case_window=self.sa
        fraud_df=self.df.copy(deep=True)
        write=self.w

        reduced_tbl_1=fraud_df.iloc[:,[0,1,3,7,8]]
        reduced_tbl_2=fraud_df.iloc[:,9:-14]
        reduced_tblset=[reduced_tbl_1,reduced_tbl_2]
        reduced_tbl=pd.concat(reduced_tblset,axis=1)
        reduced_tbl=reduced_tbl.reset_index(drop=True)

        range_oos=range(start_OOS_year,end_OOS_year+1)

        tbl_year_IS_CV=reduced_tbl.loc[np.logical_and(reduced_tbl.fyear<start_OOS_year,\
                                                   reduced_tbl.fyear>=sample_start)]
        tbl_year_IS_CV=tbl_year_IS_CV.reset_index(drop=True)
        misstate_firms=np.unique(tbl_year_IS_CV.gvkey[tbl_year_IS_CV.AAER_DUMMY==1])

        X_CV=tbl_year_IS_CV.iloc[:,-28:]

        Y_CV=tbl_year_IS_CV.AAER_DUMMY

        P_f=np.sum(Y_CV==1)/len(Y_CV)
        P_nf=1-P_f
        
        # optimize RUSBoost number of estimators
        if cv_type=='kfold':
            if cross_val==True:
            
                print('Grid search hyperparameter optimisation started for RUSBoost')
                t1=datetime.now()
                param_grid_rusboost={'n_estimators':[10,20,50,100,200,500,1000],
                                     'learning_rate':[1e-5,1e-4,1e-3,1e-2,.1,1]}
                #base_tree=DecisionTreeClassifier(min_samples_leaf=5)
                base_tree=DecisionTreeClassifier(min_samples_leaf=5)
                bao_RUSboost=RUSBoostClassifier(base_estimator=base_tree,\
                                 sampling_strategy=1,random_state=0)
                clf_rus = GridSearchCV(bao_RUSboost, param_grid_rusboost,scoring='roc_auc',\
                                   n_jobs=-1,cv=k_fold,refit=False)
                clf_rus.fit(X_CV, Y_CV)
                opt_params_rus=clf_rus.best_params_
                n_opt_rus=opt_params_rus['n_estimators']
                r_opt_rus=opt_params_rus['learning_rate']
                score_rus=clf_rus.best_score_        
                t2=datetime.now()
                dt=t2-t1
                print('RUSBoost CV finished after '+str(dt.total_seconds())+' sec')
                print('RUSBoost: The optimal number of estimators is '+str(n_opt_rus))
            else:
                n_opt_rus=200
                r_opt_rus=1e-5
                
                
                print('Cross-validation skipped ... number of estimators='+str(n_opt_rus))
        elif cv_type=='temp':
            if cross_val==True:
                cutoff_temporal=2001-temp_year
                X_CV_train=X_CV[tbl_year_IS_CV['fyear']<cutoff_temporal]
                Y_CV_train=Y_CV[tbl_year_IS_CV['fyear']<cutoff_temporal]
                X_CV_test=X_CV[tbl_year_IS_CV['fyear']>=cutoff_temporal]
                Y_CV_test=Y_CV[tbl_year_IS_CV['fyear']>=cutoff_temporal]
                
                
                print('Grid search hyperparameter optimisation started for RUSBoost')
                t1=datetime.now()
                param_grid_rusboost={'n_estimators':[10,20,50,100,200,500,1000],
                                     'learning_rate':[1e-5,1e-4,1e-3,1e-2,.1,1]}
                
                temp_rusboost={'n_estimators':[],'learning_rate':[],'score':[]}
                
                
                for n in param_grid_rusboost['n_estimators']:
                    for r in param_grid_rusboost['learning_rate']:
                        temp_rusboost['n_estimators'].append(n)
                        temp_rusboost['learning_rate'].append(r)
                        base_tree=DecisionTreeClassifier(min_samples_leaf=5)
                        bao_RUSboost=RUSBoostClassifier(base_estimator=base_tree,\
                                         sampling_strategy=1,n_estimators=n,
                                         learning_rate=r,random_state=0).fit(
                                             X_CV_train,Y_CV_train)
                        predicted_test_RUS=bao_RUSboost.predict_proba(X_CV_test)[:,-1]
                        
                        temp_rusboost['score'].append(roc_auc_score(Y_CV_test,predicted_test_RUS))
                
                idx_opt_rus=temp_rusboost['score'].index(np.max(temp_rusboost['score']))
                n_opt_rus=temp_rusboost['n_estimators'][idx_opt_rus]
                r_opt_rus=temp_rusboost['learning_rate'][idx_opt_rus]
                score_rus=temp_rusboost['score'][idx_opt_rus]
                    
                t2=datetime.now()
                dt=t2-t1
                print('RUSBoost Temporal validation finished after '+str(dt.total_seconds())+' sec')
                print('RUSBoost: The optimal number of estimators is '+str(n_opt_rus))
                
            else:
                n_opt_rus=1000
                r_opt_rus=1e-2
                #n_opt=3000
                print('Temporal-validation skipped ... number of estimators='+str(n_opt_rus))
                        
                
        # Setting as proposed in Bao et al (2020)

        roc_rusboost=np.zeros(len(range_oos))
        specificity_rusboost=np.zeros(len(range_oos))
        sensitivity_OOS_rusboost=np.zeros(len(range_oos))
        precision_rusboost=np.zeros(len(range_oos))
        sensitivity_OOS_rusboost1=np.zeros(len(range_oos))
        specificity_OOS_rusboost1=np.zeros(len(range_oos))
        precision_rusboost1=np.zeros(len(range_oos))
        ndcg_rusboost1=np.zeros(len(range_oos))
        ecm_rusboost1=np.zeros(len(range_oos))

        m=0

        for yr in range_oos:
            t1=datetime.now()
            
            year_start_IS=sample_start
            tbl_year_IS=reduced_tbl.loc[np.logical_and(reduced_tbl.fyear<yr-OOS_gap,\
                                                       reduced_tbl.fyear>=year_start_IS)]
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
            
            Y_OOS=tbl_year_OOS.AAER_DUMMY
            
            n_P=np.sum(Y_OOS==1)
            n_N=np.sum(Y_OOS==0)
            base_tree=DecisionTreeClassifier(min_samples_leaf=5)
            bao_RUSboost=RUSBoostClassifier(base_estimator=base_tree,n_estimators=n_opt_rus,\
                             learning_rate=r_opt_rus,sampling_strategy=1,random_state=0)
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
            
            FN_rusboost1=np.sum(np.logical_and(probs_oos_fraud_rusboost<cutoff_OOS_rusboost, \
                                                          Y_OOS==1))
            FP_rusboost1=np.sum(np.logical_and(probs_oos_fraud_rusboost>=cutoff_OOS_rusboost, \
                                                          Y_OOS==0))
                
            ecm_rusboost1[m]=C_FN*P_f*FN_rusboost1/n_P+C_FP*P_nf*FP_rusboost1/n_N
            
            
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

        perf_tbl_general['Roc']=str(np.round(
            np.mean(roc_rusboost)*100,2))+'% ('+\
            str(np.round(np.std(roc_rusboost)*100,2))+'%)'
                                                    
        perf_tbl_general['Sensitivity @ 1 Prc']=str(np.round(
            np.mean(sensitivity_OOS_rusboost1)*100,2))+'% ('+\
            str(np.round(np.std(sensitivity_OOS_rusboost1)*100,2))+'%)'

        perf_tbl_general['Specificity @ 1 Prc']=str(np.round(
            np.mean(specificity_OOS_rusboost1)*100,2))+'% ('+\
            str(np.round(np.std(specificity_OOS_rusboost1)*100,2))+'%)'
        

        perf_tbl_general['Precision @ 1 Prc']=str(np.round(
            np.mean(precision_rusboost1)*100,2))+'% ('+\
            str(np.round(np.std(precision_rusboost1)*100,2))+'%)'
        
        f1_score_rusboost1=2*(precision_rusboost1*sensitivity_OOS_rusboost1)/\
            (precision_rusboost1+sensitivity_OOS_rusboost1+1e-8)
        perf_tbl_general['F1 Score @ 1 Prc']=str(np.round(
            np.mean(f1_score_rusboost1)*100,2))+'% ('+\
            str(np.round(np.std(f1_score_rusboost1)*100,2))+'%)'
                                                    
        perf_tbl_general['NDCG @ 1 Prc']=str(np.round(
            np.mean(ndcg_rusboost1)*100,2))+'% ('+\
            str(np.round(np.std(ndcg_rusboost1)*100,2))+'%)'

        perf_tbl_general['ECM @ 1 Prc']=str(np.round(
            np.mean(ecm_rusboost1)*100,2))+'% ('+\
            str(np.round(np.std(ecm_rusboost1)*100,2))+'%)'
        
                   
        if cv_type=='kfold':
            lbl_perf_tbl='perf_tbl_'+str(start_OOS_year)+'_'+str(end_OOS_year)+\
                    '_'+case_window+',OOS='+str(OOS_period)+',serial='+str(adjust_serial)+\
                        ',gap='+str(OOS_gap)+'_kfold_RUSBoost.csv'
        else:
            lbl_perf_tbl='perf_tbl_'+str(start_OOS_year)+'_'+str(end_OOS_year)+\
                    '_'+case_window+',OOS='+str(OOS_period)+',serial='+str(adjust_serial)+\
                        ',gap='+str(OOS_gap)+'_temporal_RUSBoost.csv'
        if write==True:
            perf_tbl_general.to_csv(lbl_perf_tbl,index=False)
        print(perf_tbl_general)
        t_last=datetime.now()
        dt_total=t_last-t0
        print('total run time is '+str(dt_total.total_seconds())+' sec')

        # extract performance for 2003-2008 directly from 2001-2010

        perf_tbl_general=pd.DataFrame()
        perf_tbl_general['models']=['RUSBoost-28']

        perf_tbl_general['Roc']=str(np.round(
            np.mean(roc_rusboost[2:8])*100,2))+'% ('+\
            str(np.round(np.std(roc_rusboost[2:8])*100,2))+'%)'
                                                    
        perf_tbl_general['Sensitivity @ 1 Prc']=str(np.round(
            np.mean(sensitivity_OOS_rusboost1[2:8])*100,2))+'% ('+\
            str(np.round(np.std(sensitivity_OOS_rusboost1[2:8])*100,2))+'%)'

        perf_tbl_general['Specificity @ 1 Prc']=str(np.round(
            np.mean(specificity_OOS_rusboost1[2:8])*100,2))+'% ('+\
            str(np.round(np.std(specificity_OOS_rusboost1[2:8])*100,2))+'%)'
        

        perf_tbl_general['Precision @ 1 Prc']=str(np.round(
            np.mean(precision_rusboost1[2:8])*100,2))+'% ('+\
            str(np.round(np.std(precision_rusboost1[2:8])*100,2))+'%)'
        
        perf_tbl_general['F1 Score @ 1 Prc']=str(np.round(
            np.mean(f1_score_rusboost1[2:8])*100,2))+'% ('+\
            str(np.round(np.std(f1_score_rusboost1[2:8])*100,2))+'%)'
                                                    
        perf_tbl_general['NDCG @ 1 Prc']=str(np.round(
            np.mean(ndcg_rusboost1[2:8])*100,2))+'% ('+\
            str(np.round(np.std(ndcg_rusboost1[2:8])*100,2))+'%)'
        

        perf_tbl_general['ECM @ 1 Prc']=str(np.round(
            np.mean(ecm_rusboost1)*100,2))+'% ('+\
            str(np.round(np.std(ecm_rusboost1)*100,2))+'%)'

                                                   
        if cv_type=='kfold':
            lbl_perf_tbl='perf_tbl_'+str(2003)+'_'+str(2008)+\
                    '_'+case_window+',OOS='+str(OOS_period)+',serial='+str(adjust_serial)+\
                        ',gap='+str(OOS_gap)+'_kfold_RUSBoost.csv'
        else:
            lbl_perf_tbl='perf_tbl_'+str(2003)+'_'+str(2008)+\
                    '_'+case_window+',OOS='+str(OOS_period)+',serial='+str(adjust_serial)+\
                        ',gap='+str(OOS_gap)+'_temporal_RUSBoost.csv'
                        
        if write==True:
            perf_tbl_general.to_csv(lbl_perf_tbl,index=False)
        
        ## End of analysis of 28 raw variables procedure 
        

    def analyse_fk(self,C_FN=30,C_FP=1,record_matrix=True):
        """
        This code replicates the SVM with Financial Kernel as in Cecchini et al (2010)
        This code uses 23 raw financial variables and generates 1518 features out of 
        the initial variable set. 
        
        Parameters:
            – C_FN: Cost of a False Negative for ECM
            – C_FP: Cost of a False Positive for ECM
            – record_matrix: True (default) /False choice between recording the feature matrix
            and the lagged data table into a pickle file to allow further runs more quickly.
            If True a pickle file of size 900MB is going to be stored on your disk.

        Outputs: 
        Main results are stored in the table variable "perf_tbl_general" written into
        2 csv files: time period 2001-2010 and 2003-2008

        Steps:
            1. Generate lagged data. For each reported financial figure for each unique
            company the script finds the last observation and records the last amount. 
            Accordingly for the set of 23 inputs we have 46 inputs in "tbl_ratio_fk".
            2. Create the feature space based on Financial Kernel as in Cecchini et al (2010).
            The mapped feature space is stored in variable "mapped_X" with 1518 attributes.
            3. k-fold Cross-validation to find optimal C+/C- ratio (optional)
            4. Estimating the performance for each OOS period.

        Warnings: 
            – Running this code can take several hours. The generation of lagged data 
            takes ~40 mins, mapped matrix of inputs (N,1518) ~320 mins, CV ~ 40 mins,
            and main analysis ~ 120 mins. These figures are estimates based on a MacBook Pro 2017.
            – To make sure the computations are stored, you can set variable "record_matrix" to True.
            – If "record_matrix==True" the code stores a pickle file of ~ 900MB on your disk.
            
        """
        
        
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import roc_auc_score
        from extra_codes import ndcg_k
        import pickle
        
        t0=datetime.now()
        IS_period=self.ip
        k_fold=self.cv_k
        OOS_period=self.op # 1 year ahead prediction
        OOS_gap=self.og # Gap between training and testing period
        start_OOS_year=self.ts[0]
        end_OOS_year=self.ts[-1]
        sample_start=self.ss
        adjust_serial=self.a_s
        cross_val=self.cv
        cv_type=self.cv_t
        temp_year=self.cv_t_y
        case_window=self.sa
        fraud_df=self.df.copy(deep=True)
        write=self.w
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

        if isfile('features_fk.pkl')==False:
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
            tbl_ratio_fk=tbl_ratio_fk[tbl_ratio_fk.fyear>=(sample_start-1)]
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
            
        idx_CV=tbl_ratio_fk[np.logical_and(tbl_ratio_fk.fyear>=sample_start,
                                           tbl_ratio_fk.fyear<start_OOS_year)].index
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
        if cv_type=='kfold':
            if cross_val==True:
    
                print('Grid search hyperparameter optimisation started for SVM-FK')
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
        elif cv_type=='temp':
            if cross_val==True:
                
                print('Grid search hyperparameter optimisation started for SVM-FK')
                t1=datetime.now()
                cutoff_temporal=2001-temp_year
                idx_CV_train=tbl_ratio_fk[np.logical_and(tbl_ratio_fk.fyear>=1991,
                                                   tbl_ratio_fk.fyear<cutoff_temporal)].index
                X_CV_train=mapped_X[idx_CV_train]
                Y_CV_train=tbl_ratio_fk.AAER_DUMMY[idx_CV_train]
                idx_CV_test=tbl_ratio_fk[np.logical_and(tbl_ratio_fk.fyear>=cutoff_temporal,
                                                   tbl_ratio_fk.fyear<2001)].index
                X_CV_test=mapped_X[idx_CV_test]
                Y_CV_test=tbl_ratio_fk.AAER_DUMMY[idx_CV_test]
                
                param_grid_svm={'class_weight':[\
                                                {0:2e-3,1:1},{0:5e-3,1:1},
                                                {0:1e-2,1:1},{0:2e-2,1:1},{0:5e-2,1:1},\
                                                {0:1e-1,1:1},{0:2e-1,1:1},{0:5e-1,1:1},{0:1e0,1:1}]}
                
                    
                temp_svm={'class_weight':[],'score':[]}
                
                for w in param_grid_svm['class_weight']:
                    temp_svm['class_weight'].append(w)
                    
                    base_mdl_svm=SVC(shrinking=False,\
                                        probability=False,
                                        class_weight=w,\
                                        random_state=0,max_iter=-1,\
                                            tol=X_CV.shape[-1]*1e-3).fit(X_CV_train,Y_CV_train)
                    predicted_test_svc=base_mdl_svm.decision_function(X_CV_test)
                    
                    predicted_test_svc=np.exp(predicted_test_svc)/(1+np.exp(predicted_test_svc))
                    
                    temp_svm['score'].append(roc_auc_score(Y_CV_test,predicted_test_svc))
                
                idx_opt_svm=temp_svm['score'].index(np.max(temp_svm['score']))
                
                C_opt=temp_svm['class_weight'][idx_opt_svm][0]
                score_svm=temp_svm['score'][idx_opt_svm]
                opt_params_svm_fk={'class_weight':temp_svm['class_weight'][idx_opt_svm]}
                print(opt_params_svm_fk)
                t2=datetime.now()
                dt=t2-t1
                print('SVM Temporal validation finished after '+str(dt.total_seconds())+' sec')
                print('SVM: The optimal C+/C- ratio is '+str(1/C_opt))
            else:
                opt_params_svm_fk={'class_weight': {0: 0.01, 1: 1}}
                score_svm=0.702738982926179
                print('Temporal validation skipped ... Using C+/C-='+\
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

        m=0

        for yr in range_oos:
            t1=datetime.now()
            if case_window=='expanding':
                year_start_IS=sample_start
            else:
                year_start_IS=yr-IS_period
            idx_IS=tbl_ratio_fk[np.logical_and(tbl_ratio_fk.fyear<yr-OOS_gap,\
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
                
            
            t2=datetime.now() 
            dt=t2-t1
            print('analysis finished for OOS period '+str(yr)+' after '+str(dt.total_seconds())+' sec')
            m+=1

        print('average top percentile sensitivity for the period '+str(start_OOS_year)+' to '+\
              str(end_OOS_year)+' is '+str(round(np.mean(sensitivity_OOS_svm1)*100,2))+\
                  '% for SVM-FK')
        f1_score_svm1=2*(precision_svm1*sensitivity_OOS_svm1)/\
            (precision_svm1+sensitivity_OOS_svm1+1e-8)
        # create performance table now
        perf_tbl_general=pd.DataFrame()
        perf_tbl_general['models']=['SVM-FK-23']
        
        
        perf_tbl_general['Roc']=str(np.round(
            np.mean(roc_svm)*100,2))+'% ('+\
            str(np.round(np.std(roc_svm)*100,2))+'%)'
                                                    
        perf_tbl_general['Sensitivity @ 1 Prc']=str(np.round(
            np.mean(sensitivity_OOS_svm1)*100,2))+'% ('+\
            str(np.round(np.std(sensitivity_OOS_svm1)*100,2))+'%)'

        perf_tbl_general['Specificity @ 1 Prc']=str(np.round(
            np.mean(specificity_OOS_svm1)*100,2))+'% ('+\
            str(np.round(np.std(specificity_OOS_svm1)*100,2))+'%)'
        

        perf_tbl_general['Precision @ 1 Prc']=str(np.round(
            np.mean(precision_svm1)*100,2))+'% ('+\
            str(np.round(np.std(precision_svm1)*100,2))+'%)'
        
        
        perf_tbl_general['F1 Score @ 1 Prc']=str(np.round(
            np.mean(f1_score_svm1)*100,2))+'% ('+\
            str(np.round(np.std(f1_score_svm1)*100,2))+'%)'
                                                    
        perf_tbl_general['NDCG @ 1 Prc']=str(np.round(
            np.mean(ndcg_svm1)*100,2))+'% ('+\
            str(np.round(np.std(ndcg_svm1)*100,2))+'%)'

        perf_tbl_general['ECM @ 1 Prc']=str(np.round(
            np.mean(ecm_svm1)*100,2))+'% ('+\
            str(np.round(np.std(ecm_svm1)*100,2))+'%)'


        if cv_type=='kfold':
            lbl_perf_tbl='perf_tbl_'+str(start_OOS_year)+'_'+str(end_OOS_year)+\
                    '_'+case_window+',OOS='+str(OOS_period)+',serial='+str(adjust_serial)+\
                        ',gap='+str(OOS_gap)+'_kfold_SVM_FK.csv'
        else:
            lbl_perf_tbl='perf_tbl_'+str(start_OOS_year)+'_'+str(end_OOS_year)+\
                    '_'+case_window+',OOS='+str(OOS_period)+',serial='+str(adjust_serial)+\
                        ',gap='+str(OOS_gap)+'_temporal_SVM_FK.csv'

        if write==True:
            perf_tbl_general.to_csv(lbl_perf_tbl,index=False)
        print(perf_tbl_general)
        t_last=datetime.now()
        dt_total=t_last-t0
        print('total run time is '+str(dt_total.total_seconds())+' sec')


        # extract performance for 2003-2008 directly from 2001-2010

        perf_tbl_general=pd.DataFrame()
        perf_tbl_general['models']=['SVM-FK-23']

        perf_tbl_general['Roc']=str(np.round(
            np.mean(roc_svm[2:8])*100,2))+'% ('+\
            str(np.round(np.std(roc_svm[2:8])*100,2))+'%)'
                                                    
        perf_tbl_general['Sensitivity @ 1 Prc']=str(np.round(
            np.mean(sensitivity_OOS_svm1[2:8])*100,2))+'% ('+\
            str(np.round(np.std(sensitivity_OOS_svm1[2:8])*100,2))+'%)'

        perf_tbl_general['Specificity @ 1 Prc']=str(np.round(
            np.mean(specificity_OOS_svm1[2:8])*100,2))+'% ('+\
            str(np.round(np.std(specificity_OOS_svm1[2:8])*100,2))+'%)'
        

        perf_tbl_general['Precision @ 1 Prc']=str(np.round(
            np.mean(precision_svm1[2:8])*100,2))+'% ('+\
            str(np.round(np.std(precision_svm1[2:8])*100,2))+'%)'
        
        
        perf_tbl_general['F1 Score @ 1 Prc']=str(np.round(
            np.mean(f1_score_svm1[2:8])*100,2))+'% ('+\
            str(np.round(np.std(f1_score_svm1[2:8])*100,2))+'%)'
                                                    
        perf_tbl_general['NDCG @ 1 Prc']=str(np.round(
            np.mean(ndcg_svm1[2:8])*100,2))+'% ('+\
            str(np.round(np.std(ndcg_svm1[2:8])*100,2))+'%)'

        perf_tbl_general['ECM @ 1 Prc']=str(np.round(
            np.mean(ecm_svm1[2:8])*100,2))+'% ('+\
            str(np.round(np.std(ecm_svm1[2:8])*100,2))+'%)'                                  

        if cv_type=='kfold':
            lbl_perf_tbl='perf_tbl_'+str(2003)+'_'+str(2008)+\
                    '_'+case_window+',OOS='+str(OOS_period)+',serial='+str(adjust_serial)+\
                        ',gap='+str(OOS_gap)+'_kfold_SVM_FK.csv'
        else:
            lbl_perf_tbl='perf_tbl_'+str(2003)+'_'+str(2008)+\
                    '_'+case_window+',OOS='+str(OOS_period)+',serial='+str(adjust_serial)+\
                        ',gap='+str(OOS_gap)+'_temporal_SVM_FK.csv'
        if write==True:
            perf_tbl_general.to_csv(lbl_perf_tbl,index=False)
        
        ## End of analysis of 23 raw variables procedure as in Cecchini
    
    
    
    
    
    def analyse_forward(self,C_FN=30,C_FP=1):
        """
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

        Parameters:
            – C_FN: Cost of a False Negative for ECM
            – C_FP: Cost of a False Positive for ECM


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
            – Running this code can take up to 180 mins if the Financial Kernel pickle 
            file is already loaded.
            These figures are estimates based on a MacBook Pro 2021.
        
        """
        
        from scipy.stats import mannwhitneyu
        from sklearn.linear_model import SGDClassifier,LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import AdaBoostClassifier
        from imblearn.ensemble import RUSBoostClassifier
        from statsmodels.stats.weightstats import ttest_ind
        from extra_codes import relogit
        from statsmodels.discrete.discrete_model import Logit
        from statsmodels.tools import add_constant
        
        import pickle
        
        t0=datetime.now()
        # setting the parameters
        IS_period=self.ip
        k_fold=self.cv_k
        OOS_period=self.op # 1 year ahead prediction
        OOS_gap=self.og # Gap between training and testing period
        start_OOS_year=self.ts[0]
        end_OOS_year=self.ts[-1]
        sample_start=self.ss
        adjust_serial=self.a_s
        cross_val=self.cv
        case_window=self.sa
        fraud_df=self.df.copy(deep=True)
        write=self.w

        fyears_available=np.unique(fraud_df.fyear)
        count_over=count_fraud=np.zeros(fyears_available.shape)

        reduced_tbl_1=fraud_df.iloc[:,[0,1,3,7,8]]
        reduced_tbl_2=fraud_df.iloc[:,9:-3]
        reduced_tblset=[reduced_tbl_1,reduced_tbl_2]
        reduced_tbl=pd.concat(reduced_tblset,axis=1)
        reduced_tbl=reduced_tbl[reduced_tbl.fyear>=sample_start]

        tbl_fk_svm=reduced_tbl.copy()

        tbl_fk_svm.pop('act')
        tbl_fk_svm.pop('ap')
        tbl_fk_svm.pop('ppegt')
        tbl_fk_svm.pop('dltis')
        tbl_fk_svm.pop('sstk')

        # cross-validation optimised parameters used below for the ratio-based ML models
        ### RUSBOost 28
        n_opt_rus=200
        r_opt_rus=1e-5
        
        ### Ratio-based ML models
        opt_params_svm={'class_weight': {0: 0.01, 1: 1}, 'kernel': 'linear'}
        C_opt=opt_params_svm['class_weight'][0]
        kernel_opt=opt_params_svm['kernel']
        score_svm=0.701939025416111
        
        score_lr=0.7056438104977343
        
        opt_params_sgd={'class_weight': {0: 5e-3, 1: 1}, 'loss': 'log', 'penalty': 'l2'}
        score_sgd=0.7026775920776185

        opt_params_ada={'learning_rate': 0.9, 'n_estimators': 20}
        score_ada=0.700229450411913
        
        opt_params_mlp={'activation': 'logistic', 'hidden_layer_sizes': 5, 'solver': 'adam'}
        score_mlp=0.706333862286029
        

         
        # Optimised setting for Cecchini et al (2010) – SVM Kernel
        opt_params_svm_fk={'class_weight':{0: 0.02, 1: 1}}
        score_svm=0.595534973722555
        
        
        if isfile('features_fk.pkl')==True:    
            dict_db=pickle.load(open('features_fk.pkl','r+b'))
            tbl_ratio_fk=dict_db['lagged_Data']
            mapped_X=dict_db['matrix']
            red_tbl_fk=tbl_ratio_fk.iloc[:,-46:]
            print('pickle file for SVM-FK loaded successfully ...')
        else:    
            raise NameError ('The pickle file for the financial kernel missing. Rerun the SVM FK procedure first...')


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
            year_start_IS=sample_start

            # Setting the IS for all models but SVM-FK
            tbl_year_IS=reduced_tbl.loc[np.logical_and(reduced_tbl.fyear<yr,\
                                                       reduced_tbl.fyear>=year_start_IS)]
            tbl_year_IS=tbl_year_IS.reset_index(drop=True)
            misstate_firms=np.unique(tbl_year_IS.gvkey[tbl_year_IS.AAER_DUMMY==1])
            tbl_year_OOS=reduced_tbl.loc[np.logical_and(reduced_tbl.fyear>=yr,\
                                                        reduced_tbl.fyear<yr+OOS_period)]
            X=tbl_year_IS.iloc[:,-11:]
            
            
            mean_vals=np.mean(X)
            std_vals=np.std(X)
            X=(X-mean_vals)/std_vals
            
            X_rus=tbl_year_IS.iloc[:,-39:-11]
            
            
            
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
            bao_RUSboost=RUSBoostClassifier(base_estimator=base_tree,n_estimators=n_opt_rus,\
                             learning_rate=r_opt_rus,sampling_strategy=1,random_state=0)
            clf_rusboost = bao_RUSboost.fit(X_rus,Y)
            probs_oos_fraud_rus=clf_rusboost.predict_proba(X_rus_OOS)[:,-1]
            
            
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
            clf_lr = Logit(Y,X)
            clf_lr=clf_lr.fit(disp=0)
            probs_oos_fraud_lr=clf_lr.predict(X_OOS)
            
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
            clf_mlp=MLPClassifier(hidden_layer_sizes=opt_params_mlp['hidden_layer_sizes'], \
                                  activation=opt_params_mlp['activation'],solver=opt_params_mlp['solver'],\
                                               random_state=0,validation_fraction=.1)
            clf_mlp=clf_mlp.fit(X,Y)
            probs_oos_fraud_mlp=clf_mlp.predict_proba(X_OOS)[:,-1]
                        
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
            
            probs_oos_fraud_svm=(1+np.exp(-1*probs_oos_fraud_svm))**-1
                
            probs_oos_fraud_lr=(1+np.exp(-1*probs_oos_fraud_lr))**-1
                
            probs_oos_fraud_sgd=(1+np.exp(-1*probs_oos_fraud_sgd))**-1
            
            probs_oos_fraud_ada=(1+np.exp(-1*probs_oos_fraud_ada))**-1
                
            probs_oos_fraud_mlp=(1+np.exp(-1*probs_oos_fraud_mlp))**-1
            
            clf_fused=np.dot(np.array([probs_oos_fraud_svm,\
                                  probs_oos_fraud_lr,probs_oos_fraud_sgd,
                                  probs_oos_fraud_ada,\
                                      probs_oos_fraud_mlp]).T,weight_ser)
            
            probs_oos_fraud_fused=clf_fused
                        
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

        

        forward_tbl['ttest pval']=[pval_svm_fk,pval_rus,pval_svm,pval_lr,pval_sgd,\
                                         pval_ada,pval_mlp,pval_fused]

        forward_tbl['Std Forward Precision']=[np.std(precision_svm_fk1),\
                                              np.std(precision_rus1),np.std(precision_svm1),\
                                         np.std(precision_lr1),\
                                             np.std(precision_sgd1),np.std(precision_ada1),\
                                                 np.std(precision_mlp1),np.std(precision_fused1)]
        
        pval_mw_svm_fk=mannwhitneyu(precision_svm1,precision_svm_fk1,alternative='less')[1]
        pval_mw_rus=mannwhitneyu(precision_svm1,precision_rus1,alternative='less')[1]
        pval_mw_svm=mannwhitneyu(precision_svm1,precision_svm1,alternative='less')[1]
        pval_mw_lr=mannwhitneyu(precision_svm1,precision_lr1,alternative='less')[1]
        pval_mw_sgd=mannwhitneyu(precision_svm1,precision_sgd1,alternative='less')[1]
        pval_mw_ada=mannwhitneyu(precision_svm1,precision_ada1,alternative='less')[1]
        pval_mw_mlp=mannwhitneyu(precision_svm1,precision_mlp1,alternative='less')[1]
        pval_mw_fused=mannwhitneyu(precision_svm1,precision_fused1,alternative='less')[1]
        # Only LogitBoost is significant
        forward_tbl['Median Forward Precision']=[np.median(precision_svm_fk1),\
                                               np.median(precision_rus1),np.median(precision_svm1),\
                                         np.median(precision_lr1),\
                                             np.median(precision_sgd1),np.median(precision_ada1),\
                                                 np.median(precision_mlp1),np.median(precision_fused1)]    
        
        forward_tbl['mw pval']=[pval_mw_svm_fk,pval_mw_rus,pval_mw_svm,pval_mw_lr,\
                                pval_mw_sgd,pval_mw_ada,pval_mw_mlp,pval_mw_fused]

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
        
        forward_tbl['ttest pval when']=[pval_when_svm_fk,pval_when_rus,\
                                        pval_when_svm,pval_when_lr,pval_when_sgd,\
                                         pval_when_ada,pval_when_mlp,pval_when_fused]

        forward_tbl['Std Year Ahead']=[np.std(avg_early_svm_fk1),np.std(avg_early_rus1),\
                                       np.std(avg_early_svm1),np.std(avg_early_lr1),\
                                             np.std(avg_early_sgd1),np.std(avg_early_ada1),\
                                                 np.std(avg_early_mlp1),np.std(avg_early_fused1)]

        forward_tbl['Median Year Ahead']=[np.median(avg_early_svm_fk1),np.median(avg_early_rus1),\
                                        np.median(avg_early_svm1),np.median(avg_early_lr1),\
                                             np.median(avg_early_sgd1),np.median(avg_early_ada1),\
                                                 np.median(avg_early_mlp1),np.median(avg_early_fused1)]
        
        pval_mw_when_svm_fk=mannwhitneyu(avg_early_svm1,avg_early_svm_fk1,alternative='less')[1]
        pval_mw_when_rus=mannwhitneyu(avg_early_svm1,avg_early_rus1,alternative='less')[1]
        pval_mw_when_svm=mannwhitneyu(avg_early_svm1,avg_early_svm1,alternative='less')[1]
        pval_mw_when_lr=mannwhitneyu(avg_early_svm1,avg_early_lr1,alternative='less')[1]
        pval_mw_when_sgd=mannwhitneyu(avg_early_svm1,avg_early_sgd1,alternative='less')[1]
        pval_mw_when_ada=mannwhitneyu(avg_early_svm1,avg_early_ada1,alternative='less')[1]
        pval_mw_when_mlp=mannwhitneyu(avg_early_svm1,avg_early_mlp1,alternative='less')[1]
        pval_mw_when_fused=mannwhitneyu(avg_early_svm1,avg_early_fused1,alternative='less')[1]   
        # Only LogitBoost is significant
        
        forward_tbl['mw pval when']=[pval_mw_when_svm_fk,pval_mw_when_rus,\
                                     pval_mw_when_svm,pval_mw_when_lr,\
                                pval_mw_when_sgd,pval_mw_when_ada,\
                                    pval_mw_when_mlp,pval_mw_when_fused]

        
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

        if write==True:
            forward_tbl.to_csv(lbl_perf_tbl,index=False)
        print(forward_tbl)
        t_last=datetime.now()
        dt_total=t_last-t0
        print('total run time is '+str(dt_total.total_seconds())+' sec')
        
        ## End of foreward looking analysis
    
    def compare_ada(self,C_FN=30,C_FP=1):
        """
        This code uses 11 financial ratios to compare performance of an AdaBoost with 
        decision tree learners with AdaBoost with Logistic Regression

        Parameters:
            – C_FN: Cost of a False Negative for ECM
            – C_FP: Cost of a False Positive for ECM
        

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
        
        """
        
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import roc_auc_score
        from sklearn.tree import DecisionTreeClassifier
        from datetime import datetime
        from extra_codes import ndcg_k
        
        t0=datetime.now()
        # setting the parameters
        IS_period=self.ip
        k_fold=self.cv_k
        OOS_period=self.op # 1 year ahead prediction
        OOS_gap=self.og # Gap between training and testing period
        start_OOS_year=self.ts[0]
        end_OOS_year=self.ts[-1]
        sample_start=self.ss
        adjust_serial=self.a_s
        cross_val=self.cv
        case_window=self.sa
        fraud_df=self.df
        write=self.w


        fyears_available=np.unique(fraud_df.fyear)
        count_over=count_fraud=np.zeros(fyears_available.shape)

        reduced_tbl_1=fraud_df.iloc[:,[0,1,3,7,8]]
        reduced_tbl_2=fraud_df.iloc[:,-14:-3]
        reduced_tblset=[reduced_tbl_1,reduced_tbl_2]
        reduced_tbl=pd.concat(reduced_tblset,axis=1)
        reduced_tbl=reduced_tbl[reduced_tbl.fyear>=sample_start]
        reduced_tbl=reduced_tbl[reduced_tbl.fyear<=end_OOS_year]

        # Setting the cross-validation setting

        tbl_year_IS_CV=reduced_tbl.loc[np.logical_and(reduced_tbl.fyear<start_OOS_year,\
                                                   reduced_tbl.fyear>=start_OOS_year-IS_period)]
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

        if cross_val is True: 
            
            
            # optimise AdaBoost with logistic regression (Ada-LR)
            print('Grid search hyperparameter optimisation started for AdaBoost')
            t1=datetime.now()
            
            param_grid_ada={'n_estimators':[10,20,50,100,200,500],\
                            'learning_rate':[.1,5e-1,9e-1,1]}
                
            best_perf_ada=0
            
            base_lr=LogisticRegression(random_state=0,solver='newton-cg')
            base_mdl_ada=AdaBoostClassifier(base_estimator=base_lr,random_state=0)
            
            clf_ada_lr = GridSearchCV(base_mdl_ada, param_grid_ada,scoring='roc_auc',\
                               n_jobs=-1,cv=k_fold,refit=False)
            clf_ada_lr.fit(X_CV, Y_CV)
            score_ada_lr=clf_ada_lr.best_score_
            if score_ada_lr>=best_perf_ada:
                best_perf_ada=score_ada_lr
                opt_params_ada_lr=clf_ada_lr.best_params_
                
            
            t2=datetime.now()
            dt=t2-t1
            print('AdaBoost-LR CV finished after '+str(dt.total_seconds())+' sec')
            
            print('AdaBoost-LR: The optimal number of estimators is '+\
                  str(opt_params_ada_lr['n_estimators'])+', and learning rate '+\
                      str(opt_params_ada_lr['learning_rate']))
            
            
            # optimise AdaBoost with tree learners (Ada-Tree): this is the basic model    
            t1=datetime.now()
            
            best_perf_ada_tree=0
            
            base_tree=DecisionTreeClassifier(min_samples_leaf=5)
            base_mdl_ada=AdaBoostClassifier(base_estimator=base_tree,random_state=0)
            clf_ada_tree = GridSearchCV(base_mdl_ada, param_grid_ada,scoring='roc_auc',\
                           n_jobs=-1,cv=k_fold,refit=False)
            clf_ada_tree.fit(X_CV, Y_CV)
            score_ada_tree=clf_ada_tree.best_score_
            if score_ada_tree>best_perf_ada_tree:
                best_perf_ada_tree=score_ada_tree
                opt_params_ada_tree=clf_ada_tree.best_params_                
            
            t2=datetime.now()
            dt=t2-t1
            print('AdaBoost-Tree CV finished after '+str(dt.total_seconds())+' sec')
            
            print('AdaBoost-Tree: The optimal number of estimators is '+\
                  str(opt_params_ada_tree['n_estimators'])+', and learning rate '+\
                      str(opt_params_ada_tree['learning_rate']))
            
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
                year_start_IS=sample_start
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
            
                                                  
        lbl_perf_tbl='Compare_Ada_'+str(start_OOS_year)+'_'+str(end_OOS_year)+\
            '_'+case_window+',OOS='+str(OOS_period)+','+\
            str(k_fold)+'fold'+',serial='+str(adjust_serial)+'_11ratios.csv'

        if write==True:
            perf_tbl_general.to_csv(lbl_perf_tbl,index=False)
        print(perf_tbl_general)
        t_last=datetime.now()
        dt_total=t_last-t0
        print('total run time is '+str(dt_total.total_seconds())+' sec')
        
        ## End of comparative procedure for AdaBoost
    
        
    def compare_logit(self,C_FN=30,C_FP=1):
        """
        This code uses 11 financial ratios to compare performance of a regulated
        Logistic Regression as in the main text with the Rare Event Logit of
        Gary King and Langche Zeng. 2001 https://tinyurl.com/y463rgub

        Parameters:
            – C_FN: Cost of a False Negative for ECM
            – C_FP: Cost of a False Positive for ECM
        

        Predictive models:
            – Basic logit
            – Class-Balanced Logit (class-LR)
            – Rare Event Logit (RE-logit)

        Outputs: 
        Main results are stored in the table variable "perf_tbl_general" written into
        1 csv files: time period 2001-2010 

        Steps:
            1. Estimating ROC for CV for RE-logit
            2. Estimating the performance for each OOS period.

        Warnings: 
            – Running this code can take up to 60 mins. 
            These figures are estimates based on a MacBook Pro 2021.
        
        """
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GridSearchCV,train_test_split
        from datetime import datetime
        from extra_codes import ndcg_k,relogit
        from statsmodels.discrete.discrete_model import Logit
        from statsmodels.tools import add_constant
        
        
        t0=datetime.now()
        # setting the parameters
        IS_period=self.ip
        k_fold=self.cv_k
        OOS_period=self.op # 1 year ahead prediction
        OOS_gap=self.og # Gap between training and testing period
        start_OOS_year=self.ts[0]
        end_OOS_year=self.ts[-1]
        sample_start=self.ss
        adjust_serial=self.a_s
        cross_val=self.cv
        case_window=self.sa
        fraud_df=self.df
        write=self.w


        fyears_available=np.unique(fraud_df.fyear)
        count_over=count_fraud=np.zeros(fyears_available.shape)

        reduced_tbl_1=fraud_df.iloc[:,[0,1,3,7,8]]
        reduced_tbl_2=fraud_df.iloc[:,-14:-3]
        reduced_tblset=[reduced_tbl_1,reduced_tbl_2]
        reduced_tbl=pd.concat(reduced_tblset,axis=1)
        reduced_tbl=reduced_tbl[reduced_tbl.fyear>=sample_start]
        reduced_tbl=reduced_tbl[reduced_tbl.fyear<=end_OOS_year]

        # Setting the cross-validation setting

        tbl_year_IS_CV=reduced_tbl.loc[np.logical_and(reduced_tbl.fyear<start_OOS_year,\
                                                   reduced_tbl.fyear>=start_OOS_year-IS_period)]
        tbl_year_IS_CV=tbl_year_IS_CV.reset_index(drop=True)
        misstate_firms=np.unique(tbl_year_IS_CV.gvkey[tbl_year_IS_CV.AAER_DUMMY==1])

        X_CV=tbl_year_IS_CV.iloc[:,-11:]

        mean_vals=np.mean(X_CV)
        std_vals=np.std(X_CV)
        X_CV=(X_CV-mean_vals)/std_vals

        Y_CV=tbl_year_IS_CV.AAER_DUMMY

        P_f=np.sum(Y_CV==1)/len(Y_CV)
        P_nf=1-P_f
        if cross_val is True:
            print('Computing CV ROC for LR and RE-logit...')
            t1=datetime.now()
            score_lr=[]
            score_relogit=[]
            
            for m in range(0,k_fold):
                train_sample,test_sample=train_test_split(Y_CV,test_size=1/
                                                          k_fold,shuffle=False,random_state=m)
                X_train=X_CV.iloc[train_sample.index]
                X_train_LR=add_constant(X_train)
                Y_train=train_sample
                X_test=X_CV.iloc[test_sample.index]
                X_test_LR=add_constant(X_test)
                Y_test=test_sample
                
                logit_model=Logit(Y_train,X_train_LR)
                logit_model=logit_model.fit(disp=0)
                pred_LR_CV=logit_model.predict(X_test_LR)
                score_lr.append(roc_auc_score(Y_test,pred_LR_CV))
                
                relogit_mdl=relogit(Y_train,X_train,add_const=True)
                pred_relogit_cv=relogit_mdl.predict(X_test)
                score_relogit.append(roc_auc_score(Y_test,pred_relogit_cv))
                
            score_lr=np.mean(score_lr)
            score_relogit=np.mean(score_relogit)
            t2=datetime.now()
            dt=t2-t1
            print('LR & RE-logit CV finished after '+str(dt.total_seconds())+' sec')
            # optimise Logistic Regression – for a regulated Logit
            print('Grid search hyperparameter optimisation started for C-LR')
            t1=datetime.now()
            
            param_grid_c_lr={'class_weight':[{0:2e-3,1:1},{0:5e-3,1:1},{0:1e-2,1:1},{0:2e-2,1:1},{0:5e-2,1:1},\
                                            {0:1e-1,1:1},{0:2e-1,1:1},{0:5e-1,1:1},{0:1e0,1:1}]}
            base_mdl_c_lr=LogisticRegression(random_state=None)
            
            clf_c_lr = GridSearchCV(base_mdl_c_lr, param_grid_c_lr,scoring='roc_auc',\
                                n_jobs=-1,cv=k_fold,refit=False)
            clf_c_lr.fit(X_CV, Y_CV)
            opt_params_c_lr=clf_c_lr.best_params_
            score_c_lr=clf_c_lr.best_score_
            C_opt_c_lr=opt_params_c_lr['class_weight'][0]
            t2=datetime.now()
            dt=t2-t1
            print('C-LR CV finished after '+str(dt.total_seconds())+' sec')
            print('C-LR: The optimal C+/C- ratio is '+str(1/C_opt_c_lr))
        else:
            score_lr=0.7056438104977343
            score_relogit=0.7071072165311988
            opt_params_c_lr={'class_weight': {0: 0.05, 1: 1}}
            C_opt_c_lr=opt_params_c_lr['class_weight'][0]
            score_c_lr=0.701876350738009
        
            
        range_oos=range(start_OOS_year,end_OOS_year+1,OOS_period)


        roc_lr=np.zeros(len(range_oos))
        sensitivity_OOS_lr1=np.zeros(len(range_oos))
        specificity_OOS_lr1=np.zeros(len(range_oos))
        precision_lr1=np.zeros(len(range_oos))
        ndcg_lr1=np.zeros(len(range_oos))
        ecm_lr1=np.zeros(len(range_oos))

        roc_relogit=np.zeros(len(range_oos))
        sensitivity_OOS_relogit1=np.zeros(len(range_oos))
        specificity_OOS_relogit1=np.zeros(len(range_oos))
        precision_relogit1=np.zeros(len(range_oos))
        ndcg_relogit1=np.zeros(len(range_oos))
        ecm_relogit1=np.zeros(len(range_oos))
        
        roc_c_lr=np.zeros(len(range_oos))
        sensitivity_OOS_c_lr1=np.zeros(len(range_oos))
        specificity_OOS_c_lr1=np.zeros(len(range_oos))
        precision_c_lr1=np.zeros(len(range_oos))
        ndcg_c_lr1=np.zeros(len(range_oos))
        ecm_c_lr1=np.zeros(len(range_oos))

        
        
        m=0
        for yr in range_oos:
            t1=datetime.now()
            if case_window=='expanding':
                year_start_IS=sample_start
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
            
            # Logistic Regression – Dechow et al (2011)
            X_lr=add_constant(X)
            clf_lr = Logit(Y,X_lr)
            clf_lr=clf_lr.fit(disp=0)
            X_OOS_lr=add_constant(X_OOS)
            probs_oos_fraud_lr=clf_lr.predict(X_OOS_lr)

            roc_lr[m]=roc_auc_score(Y_OOS,probs_oos_fraud_lr)
            
            
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
            
            # Rare Event Logit 
            relogit_model=relogit(Y,X,add_const=False)
            probs_oos_relogit=relogit_model.predict(X_OOS)
            
            roc_relogit[m]=roc_auc_score(Y_OOS,probs_oos_relogit)
            
            cutoff_OOS_relogit=np.percentile(probs_oos_relogit,99)
            sensitivity_OOS_relogit1[m]=np.sum(np.logical_and(probs_oos_relogit>=cutoff_OOS_relogit, \
                                                         Y_OOS==1))/np.sum(Y_OOS)
            specificity_OOS_relogit1[m]=np.sum(np.logical_and(probs_oos_relogit<cutoff_OOS_relogit, \
                                                          Y_OOS==0))/np.sum(Y_OOS==0)
            precision_relogit1[m]=np.sum(np.logical_and(probs_oos_relogit>=cutoff_OOS_relogit, \
                                                         Y_OOS==1))/np.sum(probs_oos_relogit>=cutoff_OOS_relogit)
            ndcg_relogit1[m]=ndcg_k(Y_OOS,probs_oos_relogit,99)
            
            FN_relogit1=np.sum(np.logical_and(probs_oos_relogit<cutoff_OOS_relogit, \
                                                          Y_OOS==1))
            FP_relogit1=np.sum(np.logical_and(probs_oos_relogit>=cutoff_OOS_relogit, \
                                                          Y_OOS==0))
                
            ecm_relogit1[m]=C_FN*P_f*FN_relogit1/n_P+C_FP*P_nf*FP_relogit1/n_N  
            
            # Class-balanced Logistic Regression
            
            clf_c_lr = LogisticRegression(class_weight=opt_params_c_lr['class_weight'],\
                                     random_state=None).fit(X,Y)

            probs_oos_c_lr=clf_c_lr.predict_proba(X_OOS)
            probs_oos_fraud_c_lr=probs_oos_c_lr[:,-1]

            roc_c_lr[m]=roc_auc_score(Y_OOS,probs_oos_fraud_c_lr)
            
            cutoff_OOS_c_lr=np.percentile(probs_oos_fraud_c_lr,99)
            sensitivity_OOS_c_lr1[m]=np.sum(np.logical_and(probs_oos_fraud_c_lr>=cutoff_OOS_c_lr, \
                                                         Y_OOS==1))/np.sum(Y_OOS)
            specificity_OOS_c_lr1[m]=np.sum(np.logical_and(probs_oos_fraud_c_lr<cutoff_OOS_c_lr, \
                                                          Y_OOS==0))/np.sum(Y_OOS==0)
            precision_c_lr1[m]=np.sum(np.logical_and(probs_oos_fraud_c_lr>=cutoff_OOS_c_lr, \
                                                         Y_OOS==1))/np.sum(probs_oos_fraud_c_lr>=cutoff_OOS_c_lr)
            ndcg_c_lr1[m]=ndcg_k(Y_OOS,probs_oos_fraud_c_lr,99)
            
            FN_c_lr1=np.sum(np.logical_and(probs_oos_fraud_c_lr<cutoff_OOS_c_lr, \
                                                          Y_OOS==1))
            FP_c_lr1=np.sum(np.logical_and(probs_oos_fraud_c_lr>=cutoff_OOS_c_lr, \
                                                          Y_OOS==0))
                
            ecm_c_lr1[m]=C_FN*P_f*FN_c_lr1/n_P+C_FP*P_nf*FP_c_lr1/n_N
            
            
            t2=datetime.now() 
            dt=t2-t1
            print('analysis finished for OOS period '+str(yr)+' after '+str(dt.total_seconds())+' sec')
            m+=1

        print('average top percentile sensitivity for the period '+str(start_OOS_year)+' to '+\
              str(end_OOS_year)+' is '+ str(round(np.mean(sensitivity_OOS_lr1)*100,2))+\
                      '% for Logit vs '+ str(round(np.mean(sensitivity_OOS_relogit1)*100,2))+\
                          '% for RE-logit vs '+ str(round(np.mean(sensitivity_OOS_c_lr1)*100,2))+\
                              '% for C-Logit vs ')

        # create performance table now
        perf_tbl_general=pd.DataFrame()
        perf_tbl_general['models']=['Logit','RE-logit','C-Logit']
        perf_tbl_general['Roc-CV']=[score_lr,score_relogit,score_c_lr]
        perf_tbl_general['Roc-OOS']=[np.mean(roc_lr),np.mean(roc_relogit),np.mean(roc_c_lr)]

                                                    
        perf_tbl_general['Sensitivity @ 1 Prc']=[np.mean(sensitivity_OOS_lr1),\
                                             np.mean(sensitivity_OOS_relogit1),\
                                                 np.mean(sensitivity_OOS_c_lr1)]

        perf_tbl_general['Specificity @ 1 Prc']=[np.mean(specificity_OOS_lr1),\
                                             np.mean(specificity_OOS_relogit1),\
                                                 np.mean(specificity_OOS_c_lr1)]

        perf_tbl_general['Precision @ 1 Prc']=[np.mean(precision_lr1),\
                                             np.mean(precision_relogit1),\
                                                 np.mean(precision_c_lr1)]

        perf_tbl_general['F1 Score @ 1 Prc']=2*(perf_tbl_general['Precision @ 1 Prc']*\
                                              perf_tbl_general['Sensitivity @ 1 Prc'])/\
                                                ((perf_tbl_general['Precision @ 1 Prc']+\
                                                  perf_tbl_general['Sensitivity @ 1 Prc']))
        perf_tbl_general['NDCG @ 1 Prc']=[np.mean(ndcg_lr1),\
                                             np.mean(ndcg_relogit1),\
                                                 np.mean(ndcg_c_lr1)]

        perf_tbl_general['ECM @ 1 Prc']=[np.mean(ecm_lr1),\
                                         np.mean(ecm_relogit1),\
                                             np.mean(ecm_c_lr1)]

        lbl_perf_tbl='Compare_Logit_'+str(start_OOS_year)+'_'+str(end_OOS_year)+\
            '_'+case_window+',OOS='+str(OOS_period)+','+\
            str(k_fold)+'fold'+',serial='+str(adjust_serial)+'_11ratios.csv'

        if write==True:
            perf_tbl_general.to_csv(lbl_perf_tbl,index=False)
        print(perf_tbl_general)
        t_last=datetime.now()
        dt_total=t_last-t0
        print('total run time is '+str(dt_total.total_seconds())+' sec')
        ## End of comparative procedure 
