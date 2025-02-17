"""
Created by T. Nishikawa based on [FAIR V-ＶＰＰ5m記号付07-11-11.xls] of Prof. Y. Masuyama
2025.02.13
"""
import numpy as np
from coeffs import Coefficients
from balance import Balance
from scipy.optimize import minimize
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
    x = blc.x(u,beta,delta,phi)
    y = blc.y(u,beta,delta,phi)
    k = blc.k(u,beta,delta,phi)
    n = blc.n(u,beta,delta,phi)
    residual = x*x + y*y + k*k + n*n
    return residual

def cons(x):
    a = 12 - np.fabs(x[0])
    b = 30 - np.fabs(x[1])
    c = 50 - np.fabs(x[2])
    d = 50 - np.fabs(x[3])    
    return np.min([a,b,c,d])

cons = (
    {'type': 'ineq', 'fun': cons}
)

cfs = Coefficients()   # 微係数
cts = Constants()      # 要目など
blc = Balance(cts,cfs)

ut = 5       # true wind speed

# --- start ---
dat = []
for gamma_t in range(30,180,5):
    x = np.array([4.0, 5.0, -15.0, -35.0])    
    #result = minimize(objective, x0=x, method="BFGS")
    result = minimize(objective, x0=x, constraints=cons, method="L-BFGS-B")
    print("twa={0:5.2f},u={1:5.2f},beta={2:5.2f},delta={3:.2f},phi={4:.2f},res={5:.2f}".format(gamma_t,result.x[0],result.x[1],result.x[2],result.x[3],result.fun))
    dat.append([gamma_t,result.x[0],result.x[1],result.x[2],result.x[3],result.fun])

# --- plot ---
d = np.array(dat).T
plt.plot(d[0],d[1],    label="Boat speed X[m/s]")
plt.plot(d[0],d[2],    label="Leeway angle[deg]")
plt.plot(d[0],-d[3],   label="Rudder angle[deg]")
plt.plot(d[0],-d[4]/10,label="Heel angle [deg/10]")
plt.xlabel("True wind angle [deg]")
plt.ylim(0,10)
plt.legend()
plt.grid()
plt.show()
