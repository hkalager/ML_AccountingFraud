"""
This script checks the functionality of the random forest implementation in the MLFraud_module.
It initializes the ML_Fraud class with specific parameters and calls the random_forest method.

Author: Arman Hassanniakalager
"""
from MLFraud_module import ML_Fraud as mf


def _main():
    a = mf(sample_start=1991, 
           test_sample=range(2001, 2011), 
           OOS_per=1, 
           OOS_gap=0, 
           sampling="expanding",
           adjust_serial=True,
           cv_flag=True, 
           cv_k=10, 
           write=True, 
           IS_per=10
           )
    a.analyse_rf()
    pass


if __name__ == "__main__":
    _main()
