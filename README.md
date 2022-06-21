# ML_AccountingFraud

This code repository is for a project to detect accounting frauds by using machine learning techniques. For more info see Wiki tab. 


# Dataset:

The data is a merged table from COMPUSTAT (via WRDS) and SEC AAER Dataset (available at https://sites.google.com/usc.edu/aaerdataset/home). Terms of access from COMPUSTAT and AAER Dataset apply. We are prohibited from disclosing the original dataset. ALL GVKEYS in this repository are randomized and may not be treated as a primary source of data.

# Replication:

1) In a terminal run the following script to make sure necessary requirements are loaded:

  `pip install -r requirements.txt`

2) In a Python environment run the following to load the module:

  `from MLFraud_module import ML_Fraud as mf`

3) Define your settings as:

  `a = mf (PARAMETERS)` where PARAMETERS are:

  `sample_start = 1991`: Calendar year marking the start of the sample;

  `test_sample = range (2001,2011)`: testing/out-of-sample period;

  `OOS_per = 1`: out-of-sample rolling period in years;

  `OOS_gap = 0`: Gap between training and testing samples in years;

  `sampling = "expanding"`: sampling style either "expanding"/"rolling";

  `adjust_serial = True`: A boolean variable to adjust for serial frauds;

  `cv_flag = False`: A boolean variable whether to replicate the cross-validation;

  `cv_k = 10`: The number of folds (k) in the cross-validation;

  `write = True`: A boolean variable whether to write results into csv files; and

  `IS_per = 10`: Number of calendar years in case a rolling training sample is used.

4) Choose any of the following procedures:

  `a.sumstats()`: to generate summary statistics and compute variance inflation factors;
  
  `a.analyse_ratio()`: to generate classification forecasts based on 11 financial ratios;
  
  `a.analyse_raw()`: to generate classification forecasts based on 28 raw financial figures as in Bao (2020);
  
  `a.analyse_fk()`: to generate classification forecasts based on 23 raw financial figures as in Cecchini (2010);
  
  `a.compare_ada()`: to compare LogitBoost with AdaBoost as in appendix; and
  
  `a.analyse_forward()`: to generate forward-looking performance results. 

# Third party resources:
1) These scripts use free-to-access Python modules Numpy, Pandas, Statsmodels, Matplotlib, Sklearn, and Imblearn. Please make sure necessary modules are installed on your machine. 
2) The script `rusboost.py` is from an anonymous online source and I adapted the code to match my programming style.
3) The script `extra_codes.py` contains a function that was adapted from the Bao et al (2020) code repository at https://github.com/JarFraud/FraudDetection
