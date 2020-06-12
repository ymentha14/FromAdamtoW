#!/usr/bin/env python3

"""
STATISTICS
"""
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats
def student_test(x1,y1,x2,y2,verbose=False):
    
    """
    performs a t-student test on the provided data following the method described in 
    Andrade, J. M., and M. G. Estévez-Pérez. "Statistical comparison of the slopes of two regression lines: A tutorial."
    Analytica chimica acta 838 (2014): 1-12.
    
    Args:
        x1 (list of float): x-coordinates of the first group
        y1 (list of foat): y-coordinates of the first group
        x2 (list of float): x-coordinates of the second group
        y1 (list of foat): y-coordinates of the second group
        verbose (bool): whether to plot some indicative output
        """
    y1 = y1.flatten()
    y2 = y2.flatten()
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    l1 = LinearRegression().fit(x1,y1)
    l2 = LinearRegression().fit(x2,y2)
    
    # coeffs for the linear regressions
    b1 = l1.coef_[0]
    b2 = l2.coef_[0]

    pred1 = l1.predict(x1)
    pred2 = l2.predict(x2)

    def steyx(x,y):
        """ get the standard deviation for the regression"""
        x = x.flatten()
        n = x.shape[0]
        a = 1/(n-2)
        b = np.std(y)**2*n
        c = (np.cov(x,y)[0,1]*(n-1))**2
        d = np.std(x)**2*n
        return np.sqrt(a*(b-c/d))
    
    stey1 = steyx(x1,y1)
    stey2 = steyx(x2,y2)

    def gets(x,ste):
        """ get the standard deviation for the slopes"""
        n = x.shape[0]
        return ste/np.sqrt(np.std(x)**2*(n))

    s1 = gets(x1,stey1)
    s2 = gets(x2,stey2)

    diff = np.abs(b2 - b1)
    
    # pythagore relation
    sediff = np.sqrt(s1**2+s2**2)

    df = n1 + n2 - 4

    # t statistic
    t = diff/sediff

    # 2-tailed test
    p = 2*(1 - stats.t.cdf(t,df=df))
    if verbose:
        print(f"b1:{b1}")
        print(f"b2:{b2}")
        print("stey1:",stey1)
        print("stey2:",stey2)
        print("b1-b2:",diff)
        print("sediff:",sediff)
        print("t:",t)
        
    return p