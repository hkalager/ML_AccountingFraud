#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 17:39:51 2021

This code stacks smaller data files (4 in our case) into a large csv file to be used in later 
stages. You need to runthis code only if you are using our sample dataset from GitHub.


Inputs: 
    – Four csv files under name "FraudDB2020_PartX.csv" where X=1,2,3,4.
    Each csv file has up to 40k observations regarding 
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
Main results are stored in the table variable "df" written into
a single csv file entitled "FraudDB2020.csv". This csv file is going to be used
in the next steps.

Warning:
    ALL GVKEYS are randomized and may not be treated as a primary source of data. 

@author: Arman Hassanniakalager GitHub: https://github.com/hkalager
Common disclaimers apply. Subject to change at all time.

Last review: 23/07/2021
"""

import pandas as pd

df=pd.DataFrame()
for s in range(1,5):
    fl_name='FraudDB2020_Part'+str(s)+'.csv'
    new_df=pd.read_csv(fl_name)
    df=df.append(new_df)

df.to_csv('FraudDB2020.csv_',index=False)
    








# You can alternatively use the code below to split the dataset into 4 smaller 
# files
# fraud_df=pd.read_csv('FraudDB2020.csv')

# fraud_df_1=fraud_df.loc[:4e4-1,:]
# fraud_df_1.to_csv('FraudDB2020_Part1.csv',index=False)
# fraud_df_2=fraud_df.loc[4e4:8e4-1,:]
# fraud_df_2.to_csv('FraudDB2020_Part2.csv',index=False)
# fraud_df_3=fraud_df.loc[8e4:1.2e5-1,:]
# fraud_df_3.to_csv('FraudDB2020_Part3.csv',index=False)
# fraud_df_4=fraud_df.loc[1.2e5:1.6e5,:]
# fraud_df_4.to_csv('FraudDB2020_Part4.csv',index=False)

