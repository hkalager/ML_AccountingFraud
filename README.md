# ML_AccountingFraud

This code repository is for a project to detect accounting frauds by using machine learning techniques. For more info see Wiki tab. 


# Dataset:

The data is a merged table from COMPUSTAT (via WRDS) and SEC AAER Dataset (available [here](https://sites.google.com/usc.edu/aaerdataset/home)). Terms of access from COMPUSTAT and AAER Dataset apply. We are prohibited from disclosing the original dataset. ALL GVKEYS in this repository are randomized and may not be treated as a primary source of data.

# Replication:

1) In a terminal run the following script to make sure necessary requirements are loaded:

`
conda install conda-build
# create env (reads environment.yml in current directory)
conda env create -f environment.yml
# activate now
conda activate ml-fraud
# to make the package accessible you can uncomment below
conda develop /path/to/your/package
`

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

4) Choose any of the following methods:

  `a.sumstats()`: to generate summary statistics and compute variance inflation factors;
  
  `a.mc_analysis(adjust_serial=True)`: 
  
  to conduct a Monte Carlo simulation for evaluating impact of serial frauds
  `adjust_serial` can be `True`, `False`, or `"baised"`;
  
  `a.analyse_ratio()`: to generate classification forecasts based on 11 financial ratios;
  
  `a.analyse_raw()`: to generate classification forecasts based on 28 raw financial figures as in Bao (2020);
  
  `a.analyse_fk()`: to generate classification forecasts based on 23 raw financial figures as in Cecchini (2010);
  
  `a.analyse_forward()`: to generate forward-looking performance results;
  
  `a.compare_ada()`: to compare LogitBoost with AdaBoost as in appendix; and
  
  `a.compare_logit()`: to compare Logit with the Rare Event Logit as in appendix.

# Third party resources:
* These scripts use free-to-access Python modules Numpy, Pandas, Statsmodels, Matplotlib, Sklearn, and Imblearn. 
* The script `extra_codes.py` contains the class `relogit` based on Rare Event Logistic Regression of [King and Zeng (2001)](https://gking.harvard.edu/files/abs/0s-abs.shtml).
* The script `extra_codes.py` contains a function that was adapted from the [Bao et al (2020) code repository](https://github.com/JarFraud/FraudDetection).
