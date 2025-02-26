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
        
        self.waterco = self.cts.rhow * self.vb * self.vb * self.cts.l * self.cts.d / 2.0
        
        self.gamma_r = np.fmin(0.057 * np.fabs(beta),0.8)    # 舵流入角減少率
        self.alpha_r = delta - self.gamma_r * beta           # 有効舵角
        
        gtb_rad = (gamma_t+beta)*np.pi/180
        self.ua = np.sqrt( ut * ut + self.vb * self.vb + 2 * ut * self.vb * np.cos(gtb_rad) )
        self.gamma_a = np.arcsin( ut * np.sin(gtb_rad)/self.ua)*180/np.pi - beta
        self.airco   = self.cts.rhoa * self.ua * self.ua * self.cts.s / 2.0

    def __pvec(self,x):
        return [ x*x*x*x*x, x*x*x*x, x*x*x, x*x, x, 1]
    
    def hull(self,u,beta,delta,phi):
        phi_rad = phi * np.pi / 180.0
        v0 = self.v0
                
        # --- X ---
        fn = self.vb / np.sqrt(self.cts.l * self.cts.g)
        ct = np.dot(self.cfs.ct,self.__pvec(fn))
        a = np.array([self.cfs.xvv, self.cfs.xvp, self.cfs.xpp,    self.cfs.xvvvv])
        b = np.array([v0*v0,        v0*phi_rad,   phi_rad*phi_rad, v0*v0*v0*v0])
        xh0 = np.dot(a,b)
        xh  = (-ct + xh0) * self.waterco

        # --- Y ---
        a = np.array([ self.cfs.yv, self.cfs.yp, self.cfs.yvvv, self.cfs.yvvp, self.cfs.yvpp,      self.cfs.yppp])
        b = np.array([ v0,          phi_rad,     v0*v0*v0,      v0*v0*phi_rad, v0*phi_rad*phi_rad, phi_rad*phi_rad*phi_rad])
        yh0 = np.dot(a,b)
        yh  = yh0 * self.waterco

        # --- K ---
        a = np.array([ self.cfs.kv, self.cfs.kp, self.cfs.kvvv, self.cfs.kvvp, self.cfs.kvpp,      self.cfs.kppp])
        b = np.array([ v0,          phi_rad,     v0*v0*v0,      v0*v0*phi_rad, v0*phi_rad*phi_rad, phi_rad*phi_rad*phi_rad])
        kh0 = np.dot(a,b)
        kh = kh0 * self.waterco * self.cts.d
        
        kh_heel = [0,0,-self.cts.disp * self.cts.g * self.cts.gm * np.sin(phi_rad),0]

        # --- N ---
        a = np.array([ self.cfs.nv, self.cfs.np, self.cfs.nvvv, self.cfs.nvvp, self.cfs.nvpp,      self.cfs.nppp])
        b = np.array([ v0,          phi_rad,     v0*v0*v0,      v0*v0*phi_rad, v0*phi_rad*phi_rad, phi_rad*phi_rad*phi_rad])
        nh0 = np.dot(a,b)
        nh = nh0 * self.waterco * self.cts.l

        return np.array([xh, yh, kh, nh]) + np.array(kh_heel)
    
    def rudder(self,u,beta,delta,phi):
        phi_rad = phi * np.pi / 180.0
        xd  = self.cfs.cxd * np.sin(self.alpha_r*np.pi/180) * np.sin(delta*np.pi/180) * self.waterco 
        yd  = self.cfs.cyd * np.sin(self.alpha_r*np.pi/180) * np.cos(delta*np.pi/180) * np.cos(phi_rad) * self.waterco
        kd  = self.cfs.ckd * np.sin(self.alpha_r*np.pi/180) * np.cos(delta*np.pi/180) * self.waterco * self.cts.d
        nd  = self.cfs.cnd * np.sin(self.alpha_r*np.pi/180) * np.cos(delta*np.pi/180) * np.cos(phi_rad) * self.waterco * self.cts.l
        return np.array([xd, yd, kd, nd])

    def sail(self,u,beta,delta,phi):
        phi_rad = phi * np.pi / 180.0
        # --- X ---        
        xs0 = np.dot(self.cfs.cxs,self.__pvec(self.gamma_a))
        xs  = xs0 * np.power(np.cos(phi_rad),2) * self.airco
        #self.xs0 = xs0
        # --- Y ---
        ys0 = -np.dot(self.cfs.cys,self.__pvec(self.gamma_a))
        ys  = ys0 * np.power(np.cos(phi_rad),2) * self.airco
        # --- K ---        
        ks0 = -np.dot(self.cfs.cks,self.__pvec(self.gamma_a))
        ks  = ks0 * np.cos(phi_rad) * self.airco * np.sqrt(self.cts.s)
        # --- N ---
        ns0 = np.dot(self.cfs.cns,self.__pvec(self.gamma_a))
        ns  = (ns0 + xs0 * (self.cts.zce/np.sqrt(self.cts.s)) * np.sin(phi_rad)) * np.power(np.cos(phi_rad),2) * self.airco * np.sqrt(self.cts.s)

        return np.array([xs, ys, ks, ns])

