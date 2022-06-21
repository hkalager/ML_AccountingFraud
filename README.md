# ML_AccountingFraud

This code repository is for a project to detect accounting frauds by using machine learning techniques. A preprint will be released soon. For more info see Wiki tab. 


# Dataset:

The data is a merged table from COMPUSTAT (via WRDS) and SEC AAER Dataset (available at https://sites.google.com/usc.edu/aaerdataset/home). Terms of access from COMPUSTAT and AAER Dataset apply. We are prohibited from disclosing the original dataset. ALL GVKEYS in this repository are randomized and may not be treated as a primary source of data.

# Replication:

Run the scripts named `step0_XX.py`, `step1_XX.py`, and so on. The first script stacks the csv files together to create a comprehensive data file. 

# Third party resources:
1) These scripts use free-to-access Python modules Numpy, Pandas, Statsmodels, Matplotlib, Sklearn, and Imblearn. Please make sure necessary modules are installed on your machine. 
2) The script `rusboost.py` is from an anonymous online source and I adapted the code to match my programming style.
3) The script `extra_codes.py` contains a function that was adapted from the Bao et al (2020) code repository at https://github.com/JarFraud/FraudDetection
