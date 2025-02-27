import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from coeffs import Coefficients
from balance import Balance
from main import Constants
from scipy.interpolate import RegularGridInterpolator

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
    blc = Balance(cts,cfs)

    delta = 0.0
    gamma_t = 90

    phi_range  = np.linspace(0,  30.0, 4)
    beta_range = np.linspace(0,   0.6, 4)
    fn_range   = np.linspace(0.1, 1.0, 10)

    hull = []
    for phi in phi_range:
        for beta in beta_range:
            for fn in fn_range:
                u = fn * np.sqrt(cts.lwl*cts.g)
                blc.update_params(u,beta,delta,phi,gamma_t,ut)
                h = blc.hull(u,beta,delta,phi)
                r = blc.rudder(u,beta,delta,phi)
                s = blc.sail(u,beta,delta,phi)
                h1 = h[0] / blc.waterco
                h2 = h[1] / blc.waterco
                h3 = h[2] / blc.waterco / cts.draft
                h4 = h[3] / blc.waterco / cts.lwl
                hull.append( np.array( [phi,beta,fn,h1,h2,h3,h4] ) )
                
            
    hull = np.array(hull)
    #print("phi, beta, fn, cfx, cfy, cmx, cmz")
    #for h in hull:
    #    print("{0:5.2f},{1:5.2f},{2:5.2f},{3:10.2e},{4:10.2e},{5:10.2e},{6:10.2e}".format(h[0],h[1],h[2],h[3],h[4],h[5],h[6]))

    # --- convert ---
    data = np.zeros(len(phi_range)*len(beta_range)*len(fn_range)*7)
    data = data.reshape(len(phi_range),len(beta_range),len(fn_range),7)
    n = 0
    for i in range(len(phi_range)):
        for j in range(len(beta_range)):
            for k in range(len(fn_range)):
                data[i][j][k] = hull[n]
                n += 1
    np.save("hull",data)

    # -------------------------
    hull = np.load("hull.npy")    
    cfx = RegularGridInterpolator((phi_range, beta_range, fn_range), hull[:,:,:,3], bounds_error=False, fill_value=None)
    cfy = RegularGridInterpolator((phi_range, beta_range, fn_range), hull[:,:,:,4], bounds_error=False, fill_value=None)                
    cmx = RegularGridInterpolator((phi_range, beta_range, fn_range), hull[:,:,:,5], bounds_error=False, fill_value=None)                
    cmz = RegularGridInterpolator((phi_range, beta_range, fn_range), hull[:,:,:,6], bounds_error=False, fill_value=None)                

    phi  = -21.0
    beta = 3.1
    fn   = 0.35
    v0 = cfx((phi, beta, fn))
    v1 = cfy((phi, beta, fn))
    v2 = cmx((phi, beta, fn))
    v3 = cmz((phi, beta, fn))
    print(v0,v1,v2,v3)        
