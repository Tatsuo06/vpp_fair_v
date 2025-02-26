"""
Created by T. Nishikawa based on [FAIR V-ＶＰＰ5m記号付07-11-11.xls] of Prof. Y. Masuyama
2025.02.13
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from vpp import *
from coeffs import *

class Constants:
    def __init__(self):
        self.rhoa = 1.2
        self.rhow = 1025.0
        self.g    = 9.807

        a = 1.0 # 2.515
        self.l    = 8.55 * a
        self.d    = 1.94 * a
        self.s    = 56.4 * a**2
        self.zce  = -6.4 * a
        self.disp = 3775.0 * a**3
        self.gm   = 1.31 * a

if __name__ == "__main__":
    ut = 5                # true wind speed
    cfs = Coefficients()
    cts = Constants()
    vpp = VPP(ut,cts,cfs)
    vpp.run(10,180,5)      # true wind angle range
    vpp.plot_polar()
    vpp.plot()
    
