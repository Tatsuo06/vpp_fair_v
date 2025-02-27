"""
Created by T. Nishikawa based on [FAIR V-ＶＰＰ5m記号付07-11-11.xls] of Prof. Y. Masuyama
2025.02.13
"""
import numpy as np

class Balance:

    def __init__(self,cts,cfs):
        self.cts = cts
        self.cfs = cfs

    def update_params(self,u,beta,delta,phi,gamma_t,ut):
        self.v  = -u * np.tan(beta*np.pi/180)  # 横流れ速度
        self.v0 =     -np.sin(beta*np.pi/180)  # 基準横流れ速度
        self.vb =  u / np.cos(beta*np.pi/180)  # 合成速度
               
        self.gamma_r = np.fmin(0.057 * np.fabs(beta),0.8)    # 舵流入角減少率
        self.alpha_r = delta - self.gamma_r * beta           # 有効舵角
        
        gtb_rad = (gamma_t+beta)*np.pi/180                   # 合成角度
        self.ua = np.sqrt( ut**2 + self.vb**2 + 2 * ut * self.vb * np.cos(gtb_rad) )
        self.gamma_a = np.arcsin( ut * np.sin(gtb_rad)/self.ua)*180/np.pi - beta

        self.waterco = 0.5 * self.cts.rhow * self.vb**2 * self.cts.wsa
        self.airco   = 0.5 * self.cts.rhoa * self.ua**2 * self.cts.sail_area

    def __pvec(self,x):
        return [ x*x*x*x*x, x*x*x*x, x*x*x, x*x, x, 1]
    
    def hull(self,u,beta,delta,phi):
        phi_rad = phi * np.pi / 180.0
        a = np.array([self.v0**2,   self.v0*phi_rad,   phi_rad**2, self.v0**4])
        b = np.array([ self.v0,          phi_rad,     self.v0**3,     self.v0**2*phi_rad, self.v0*phi_rad**2, phi_rad**3])
                
        # --- X ---
        fn = self.vb / np.sqrt(self.cts.lwl * self.cts.g)
        ct = np.dot(self.cfs.ct,self.__pvec(fn))
        xh0 = np.dot(self.cfs.hull_x,a)
        xh  = (-ct + xh0) * self.waterco

        # --- Y ---
        yh0 = np.dot(self.cfs.hull_y,b)
        yh  = yh0 * self.waterco

        # --- K ---
        kh0 = np.dot(self.cfs.hull_k,b)
        kh = kh0 * self.waterco * self.cts.draft
        
        kh_heel = [0,0,-self.cts.disp * self.cts.g * self.cts.gm * np.sin(phi_rad),0]

        # --- N ---
        nh0 = np.dot(self.cfs.hull_n,b)
        nh = nh0 * self.waterco * self.cts.lwl

        return np.array([xh, yh, kh, nh]) + np.array(kh_heel)
    
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
