#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 12:22:20 2021

This code initiates the analysis process by ensuring existing of necessary 
resources. 

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


Outputs: 
    – Summary statistics are recorded in a Pandas dataframe called "sum_Stat_tbl" 
    and is written to a csv file.
    – Variance information factor results are stored in:
        + "vif_ratios" for the 11 ratios of Dechow et al 2011
        + "reduced_tbl_raw23" for the 23 raw variables of Cecchini et al 2010
        + "reduced_tbl_raw28" for the 28 raw variables of Bao et al 2020
    – Analysis of serial frauds stored in a Pandas dataframe called "ser_tbl"
    – Figure 1: Count of frauds between 1970 to 2020
    – Figure 2: Comparison of total fraud cases vs serial frauds

Steps:
    1. Checking necessary packages and the dataset file. 
    2. Calculating Variance inflation factor for all three sets of variables:
        28 raw variables in Bao et al (2020), 23 raw variables of Cecchini 
        et al (2010), and 11 ratio variables of Dechow et al (2011)
    3. Generating summary statistics. 
    4. Calculate the proportion of serial frauds from AAERs per annum

Warnings: 
    – Running this code can take up to 1 mins. 
    These figures are estimates based on a MacBook Pro 2021.

@author: Arman Hassanniakalager GitHub: https://github.com/hkalager
Common disclaimers apply. Subject to change at all time.

Last review: 14/06/2022
"""
from extra_codes import calc_vif
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import kpss
import warnings
warnings.filterwarnings("ignore")

t0=datetime.now()
print('necessary modules exist and loaded')

fraud_df=pd.read_csv('FraudDB2020.csv')
print('necessary dataset exist and loaded')

fyears_available=[s for s in range(1991,2010+1)]

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

fig, ax = plt.subplots()
X_axis=pd.to_datetime(fyears_available,format='%Y')
ax.plot(X_axis,count_fraud,'x:b',label='AAERs')
ax.set_xlabel('Calendar Year')
ax.set_ylabel('# fraud')
ax.set_title('Count of fraud')
ax.legend()

fraud_df=fraud_df[fraud_df.fyear>=1991]
last_year=np.max(fraud_df.fyear)
fraud_df=fraud_df.reset_index(drop=True)
num_comp=len(np.unique(fraud_df.gvkey))
print(str(num_comp)+' unique in the dataset between 1991 and '+str(last_year))


reduced_tbl_ratio=fraud_df.iloc[:,-14:-3]
reduced_tbl_ratio=(reduced_tbl_ratio-np.mean(reduced_tbl_ratio))/np.std(reduced_tbl_ratio)
vif_ratios=calc_vif(reduced_tbl_ratio)
vif_ratios.to_csv('VIF_11ratio.csv',index=False)

reduced_tbl_raw28=fraud_df.iloc[:,9:-14]
reduced_tbl_raw28=(reduced_tbl_raw28-np.mean(reduced_tbl_raw28))/np.std(reduced_tbl_raw28)
vif_raw28=calc_vif(reduced_tbl_raw28)
vif_raw28.to_csv('VIF_28raw.csv',index=False)


reduced_tbl_raw23=fraud_df.iloc[:,9:-14]
reduced_tbl_raw23.pop('act')
reduced_tbl_raw23.pop('ap')
reduced_tbl_raw23.pop('ppegt')
reduced_tbl_raw23.pop('dltis')
reduced_tbl_raw23.pop('sstk')
reduced_tbl_raw23=(reduced_tbl_raw23-np.mean(reduced_tbl_raw23))/np.std(reduced_tbl_raw23)
vif_raw23=calc_vif(reduced_tbl_raw23)
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
    
sum_Stat_tbl.to_csv('SumStats.csv',index=False)
print('Summary statistics generated successfully ... ')


start_OOS_year=2001
end_OOS_year=2010

range_oos=range(start_OOS_year,end_OOS_year+1)
serial_fraud_count=np.zeros(len(range_oos))
fraud_count=np.zeros(len(range_oos))
m=0

for yr in range_oos:
    tbl_year_IS=fraud_df.loc[np.logical_and(fraud_df.fyear<yr,\
                                               fraud_df.fyear>=yr-10)]
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
ser_tbl['Year']=range_oos
ser_tbl['Fraud Case']=fraud_count
ser_tbl['Serial Fraud Case']=serial_fraud_count
ser_tbl['Serial over Total']=serial_fraud_count/fraud_count

ser_tbl.to_csv('SerialStats.csv',index=False)
print('Serial fraud results generated successfully ... ')

fig, ax = plt.subplots()
ax.plot(range_oos,serial_fraud_count,'s:r',label='# serial fraud')
ax.plot(range_oos,fraud_count,'s:b',label='# total fraud')
ax.set_xlabel('Calendar Year')
ax.set_ylabel('# firms')
ax.set_title('companies with previous history of misstatement')
ax.legend()
print('graphics generated successfully ... ')
run_time=datetime.now()-t0
print('Total runtime is '+str(run_time.total_seconds())+' seconds')
