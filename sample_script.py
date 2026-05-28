# -*- coding: utf-8 -*-
"""
This is a sample script file for testing the MLFraud_module. 
You can run this script to see how the module works and to test its functionality. 

You can modify the code as needed to fit your specific use case.

(c) Arman Hassanniakalager
"""

from MLFraud_module import ML_Fraud as mf


def _main():
    # a=mf()
    # a.mc_analysis(adjust_serial='biased')
    # a.mc_analysis(adjust_serial=True)
    # a.mc_analysis(adjust_serial=False)

    a = mf(cv_flag=False, adjust_serial=True)
    # a.analyse_raw()
    a.analyse_ratio()
    a.analyse_fk()

    # a=mf(cv_type='temp',cv_flag=False,adjust_serial=True)
    # a.analyse_raw()
    # a.analyse_ratio()
    # a.analyse_fk()

    # a=mf(cv_type='temp',cv_flag=False,adjust_serial=False)
    # a.analyse_raw()
    # a.analyse_ratio()
    # a.analyse_fk()

    # a=mf(cv_type='temp',cv_flag=False,adjust_serial=False)
    # a.analyse_raw()
    # a.analyse_ratio()
    # a.analyse_fk()

    # for gap in [1,2]:
    #     a=mf(OOS_gap=gap)
    #     a.analyse_ratio()
    #     a.analyse_raw()
    #     a.analyse_fk()
    pass


if __name__ == "__main__":
    _main()
