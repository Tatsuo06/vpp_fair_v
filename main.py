"""
Created by T. Nishikawa based on [FAIR V-ＶＰＰ5m記号付07-11-11.xls] of Prof. Y. Masuyama
2025.02.13
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from vpp4d import VPP4D
from vpp3d import VPP3D
from coeffs import Coefficients
from balance import Balance
from balance_mod import BalanceMod

class Constants:
    def __init__(self):
        self.rhoa = 1.2
        self.rhow = 1025.0
        self.g    = 9.807

        a = 1 # 2.515
        self.lwl       = 8.55 * a
        self.draft     = 1.94 * a
        self.sail_area = 56.4 * a**2
        self.zce       = -6.4 * a
        self.disp      = 3775.0 * a**3
        self.gm        = 1.31 * a

        self.wsa = self.lwl * self.draft
        
        print("LWL:       {0:.2f}".format(self.lwl))
        print("Draft:     {0:.2f}".format(self.draft))
        print("Disp:      {0:.2f}".format(self.disp))
        print("WSA:       {0:.2f}".format(self.wsa))
        print("Sail Area: {0:.2f}".format(self.sail_area))
        print("CEz:       {0:.2f}".format(self.zce))
        print("GM:        {0:.2f}".format(self.gm))
        
if __name__ == "__main__":
    ut = 5                # true wind speed
    cfs = Coefficients()
    cts = Constants()
    blc = BalanceMod(cts,cfs)
    #blc = Balance(cts,cfs)

    vpp = VPP4D(ut,cts,cfs,blc)
    vpp.run(20,180,True)      # true wind angle range
    vpp.plot_polar(0,15,"polar4dm.png")
    vpp.plot(-10,10,"result4dm.png")

    vpp = VPP3D(ut,cts,cfs,blc)
    vpp.run(20,180,True)      # true wind angle range
    vpp.plot_polar(0,15,"polar3dm.png")
    vpp.plot(-10,10,"result3dm.png")
    
