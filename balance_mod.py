"""
Created by T. Nishikawa based on [FAIR V-ＶＰＰ5m記号付07-11-11.xls] of Prof. Y. Masuyama
2025.02.13
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

class BalanceMod:

    def __init__(self,cts,cfs):
        self.cts = cts
        self.cfs = cfs

        self.hull_coeffs = np.load("hull.npy")
        rphi  = np.linspace(0,  30.0, 4)
        rbeta = np.linspace(0,   0.6, 4)
        rfn   = np.linspace(0.1, 0.5, 10)
        self.param_range = (rphi, rbeta, rfn)
        
    def update_params(self,u,beta,delta,phi,gamma_t,ut):
        self.v  = -u * np.tan(beta*np.pi/180)  # 横流れ速度
        self.v0 =     -np.sin(beta*np.pi/180)  # 基準横流れ速度
        self.vb =  u / np.cos(beta*np.pi/180)  # 合成速度
               
        self.gamma_r = np.fmin(0.057 * np.fabs(beta),0.8)    # 舵流入角減少率
        self.alpha_r = delta - self.gamma_r * beta           # 有効舵角
        
        gtb_rad = (gamma_t+beta)*np.pi/180                   # 合成角度
        self.ua = np.sqrt( ut**2 + self.vb**2 + 2 * ut * self.vb * np.cos(gtb_rad) ) # AWS
        self.gamma_a = np.arcsin( ut * np.sin(gtb_rad)/self.ua)*180/np.pi - beta     # AWA

        self.waterco = 0.5 * self.cts.rhow * self.vb**2 * self.cts.wsa
        self.airco   = 0.5 * self.cts.rhoa * self.ua**2 * self.cts.sail_area

        self.fn = u / np.sqrt(self.cts.lwl * self.cts.g)
                
    def __pvec(self,x):
        return [ x*x*x*x*x, x*x*x*x, x*x*x, x*x, x, 1]
    
    def hull(self,u,beta,delta,phi):
        cfx = RegularGridInterpolator(self.param_range, self.hull_coeffs[:,:,:,3], bounds_error=False, fill_value=None)
        cfy = RegularGridInterpolator(self.param_range, self.hull_coeffs[:,:,:,4], bounds_error=False, fill_value=None)
        cmx = RegularGridInterpolator(self.param_range, self.hull_coeffs[:,:,:,5], bounds_error=False, fill_value=None)
        cmz = RegularGridInterpolator(self.param_range, self.hull_coeffs[:,:,:,6], bounds_error=False, fill_value=None)
        xh = cfx((phi, beta, self.fn)) * self.waterco
        yh = cfy((phi, beta, self.fn)) * self.waterco
        kh = cmx((phi, beta, self.fn)) * self.waterco * self.cts.draft
        nh = cmz((phi, beta, self.fn)) * self.waterco * self.cts.lwl
        return np.array([xh, yh, kh, nh])
    
    def rudder(self,u,beta,delta,phi):
        phi_rad = phi * np.pi / 180.0
        xd  = self.cfs.rudder[0] * np.sin(self.alpha_r*np.pi/180) * np.sin(delta*np.pi/180) * self.waterco 
        yd  = self.cfs.rudder[1] * np.sin(self.alpha_r*np.pi/180) * np.cos(delta*np.pi/180) * np.cos(phi_rad) * self.waterco
        kd  = self.cfs.rudder[2] * np.sin(self.alpha_r*np.pi/180) * np.cos(delta*np.pi/180) * self.waterco * self.cts.draft
        nd  = self.cfs.rudder[3] * np.sin(self.alpha_r*np.pi/180) * np.cos(delta*np.pi/180) * np.cos(phi_rad) * self.waterco * self.cts.lwl
        return np.array([xd, yd, kd, nd])

    def sail(self,u,beta,delta,phi):
        phi_rad = phi * np.pi / 180.0
        cos_phi = np.cos(phi_rad)
        # --- X ---        
        xs0 = np.dot(self.cfs.cxs,self.__pvec(self.gamma_a))
        xs  = xs0 * cos_phi**2 * self.airco
        # --- Y ---
        ys0 = -np.dot(self.cfs.cys,self.__pvec(self.gamma_a))
        ys  = ys0 * cos_phi**2 * self.airco
        # --- K ---        
        ks0 = -np.dot(self.cfs.cks,self.__pvec(self.gamma_a))
        ks  = ks0 * cos_phi * self.airco * np.sqrt(self.cts.sail_area)
        # --- N ---
        ns0 = np.dot(self.cfs.cns,self.__pvec(self.gamma_a))
        ns1  = ns0 + xs0 * self.cts.zce / np.sqrt(self.cts.sail_area) * np.sin(phi_rad)
        ns  = ns1 * cos_phi**2 * self.airco * np.sqrt(self.cts.sail_area)

        return np.array([xs, ys, ks, ns])
