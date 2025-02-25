"""
Created by T. Nishikawa based on [FAIR V-ＶＰＰ5m記号付07-11-11.xls] of Prof. Y. Masuyama
2025.02.13
"""
import numpy as np
from coeffs import Coefficients
from balance import Balance
import scipy as sp
import matplotlib.pyplot as plt

class Constants:
    def __init__(self):
        self.rhoa = 1.2
        self.rhow = 1025.0
        self.g    = 9.807

        self.l    = 8.55
        self.d    = 1.94
        self.s    = 56.4
        self.zce  = -6.4
        self.disp = 3775.0
        self.gm   = 1.31

def objective(x):
    u     = x[0] # x方向速度
    beta  = x[1] # Leeway
    delta = x[2] # rudder angle
    phi   = x[3] # heel angle
    blc.update_params(u,beta,delta,phi,gamma_t,ut)    
    f = np.zeros(4, dtype=np.float64)
    f[0] = blc.x(u,beta,delta,phi)
    f[1] = blc.y(u,beta,delta,phi)
    f[2] = blc.k(u,beta,delta,phi)
    f[3] = blc.n(u,beta,delta,phi)
    return f
    
if __name__ == "__main__":
    ut = 5       # true wind speed

    cfs = Coefficients()   # 微係数
    cts = Constants()      # 要目など
    blc = Balance(cts,cfs)
    
    # --- start ---
    dat = [] 
    x0 = [4.0, 5.0, -15, -35]    
    for gamma_t in range(30,180,5):
        result = sp.optimize.root(objective, x0, method='lm') #'hybr'
        print("twa={0:5.2f},u={1:5.2f},beta={2:5.2f},delta={3:.2f},phi={4:.2f},residual={5:g}".format(gamma_t,result.x[0],result.x[1],result.x[2],result.x[3],np.linalg.norm(result.fun, ord=2)))
        dat.append([gamma_t,result.x[0],result.x[1],result.x[2],result.x[3]])
        
    # --- plot ---
    d = np.array(dat).T
    plt.plot(d[0],d[1],   label="Boat speed X[m/s]")
    plt.plot(d[0],d[2],   label="Leeway angle[deg]")
    plt.plot(d[0],d[3],   label="Rudder angle[deg](luff up: plus)")
    plt.plot(d[0],d[4]/10,label="Heel angle [deg/10](anti heel: plus)")
    plt.xlabel("True wind angle [deg]")
    #plt.ylim(-10,10)
    plt.legend()
    plt.grid()
    plt.show()

