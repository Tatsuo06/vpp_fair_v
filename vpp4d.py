"""
Created by T. Nishikawa based on [FAIR V-ＶＰＰ5m記号付07-11-11.xls] of Prof. Y. Masuyama
2025.02.13
"""

import numpy as np
from coeffs import Coefficients
import scipy as sp
import matplotlib.pyplot as plt

class VPP4D:
    def __init__(self,ut,cts,cfs,blc):
        self.ut  = ut
        self.cts = cts
        self.cfs = cfs
        self.blc = blc

    def run(self,a0,a1,verbose=False):
        # --- start ---
        dat = [] 
        x = [0.8*self.ut, 0, 0, 0]
        for self.gamma_t in range(a0,a1,5):
            result = sp.optimize.root(self.objective, x, method='lm') #'hybr'
            if verbose:
                print("twa={0:5.1f}, u={1:5.2f}, beta={2:5.2f}, delta={3:5.1f}, phi={4:5.1f}, residual={5:5.2f}".format(self.gamma_t,result.x[0],result.x[1],result.x[2],result.x[3],np.linalg.norm(result.fun, ord=2)))
            dat.append([self.gamma_t,result.x[0],result.x[1],result.x[2],result.x[3]])
        self.dat = np.array(dat).T
        
    def plot_polar(self,rmin,rmax,fname):    
        ax = plt.subplot(111, projection="polar")
        ax.plot(self.dat[0]*np.pi/180, self.dat[1]*3600/1852)
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")
        ax.set_title("Boat Speed[knot](4DOF)")
        ax.set_rlim([rmin, rmax])
        #plt.show()
        plt.savefig(fname)
        plt.clf()
        plt.close()

    def plot(self,ymin,ymax,fname):
        a = np.sqrt(self.cts.lwl*self.cts.g)
        plt.plot(self.dat[0], self.dat[1]/a*10, label="Boat speed Fn*10")
        plt.plot(self.dat[0], self.dat[2],      label="Leeway angle[deg]")
        plt.plot(self.dat[0], self.dat[3],      label="Rudder angle[deg](luff up: plus)")
        plt.plot(self.dat[0], self.dat[4]/10,   label="Heel angle [deg/10](anti heel: plus)")
        plt.xlabel("True wind angle [deg]")
        plt.ylim(ymin,ymax)
        plt.legend()
        plt.grid()
        plt.title("4DOF")
        #plt.show()
        plt.savefig(fname)
        plt.clf()
        plt.close()        

    def objective(self,x):
        u     = x[0] # x方向速度
        beta  = x[1] # Leeway
        delta = x[2] # rudder angle
        phi   = x[3] # heel angle
        self.blc.update_params(u,beta,delta,phi,self.gamma_t,self.ut)
        h = self.blc.hull(u,beta,delta,phi)
        r = self.blc.rudder(u,beta,delta,phi)
        s = self.blc.sail(u,beta,delta,phi)

        t = h + r + s                
        for i in range(len(t)):            
            t[i] *= t[i]

        return t
