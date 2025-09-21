from MLFraud_module import ML_Fraud as mf


def _main():
    a = mf(sample_start=1991, test_sample=range(2001, 2011), OOS_per=1, OOS_gap=0, sampling="expanding",
           adjust_serial=True,
           cv_flag=True, cv_k=10, write=True, IS_per=10)
    a.random_forest()
    pass


if __name__ == "__main__":
    _main()
