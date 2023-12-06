# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from MLFraud_module import ML_Fraud as mf

#a=mf()
# a.mc_analysis(adjust_serial='biased')
# a.mc_analysis(adjust_serial=True)
# a.mc_analysis(adjust_serial=False)

a=mf(cv_type='kfold',cv_flag=False,adjust_serial=True)
# a.analyse_raw()
a.analyse_ratio()
# a.analyse_fk()

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
