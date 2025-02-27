import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from coeffs import Coefficients
from balance import Balance
from main import Constants
from scipy.interpolate import RegularGridInterpolator

def plot_3d(n,zlabel):
    hull = np.load("hull.npy")
    gphi, gbeta, gfn = np.meshgrid(phi_range, beta_range, fn_range, indexing='ij')
    col = plt.get_cmap("hsv")
    fig = plt.figure(figsize=(12,6))

    ax1 = fig.add_subplot(121, projection='3d')
    for i in range(len(beta_range)):
        ax1.plot_wireframe( gphi[:,i,:], gfn[:,i,:], hull[:,i,:,n], color=col(i/len(beta_range)), label="beta={0:g}".format(beta_range[i]))
    ax1.legend()
    ax1.set_xlabel("phi")
    ax1.set_ylabel("fn")
    ax1.set_zlabel(zlabel)

    ax2 = fig.add_subplot(122, projection='3d')
    for i in range(len(phi_range)):
        ax2.plot_wireframe( gbeta[i,:,:], gfn[i,:,:], hull[i,:,:,n], color=col(i/len(phi_range)), label="phi={0:g}".format(phi_range[i]))
    ax2.legend()
    ax2.set_xlabel("beta")
    ax2.set_ylabel("fn")
    ax2.set_zlabel(zlabel)

    plt.savefig(zlabel+".png")
    #plt.show()
    plt.clf()
    plt.close()        

if __name__ == "__main__":
    ut = 5                # true wind speed
    cfs = Coefficients()
    cts = Constants()
    blc = Balance(cts,cfs)

    delta = 0.0
    gamma_t = 90

    phi_range  = np.linspace(0,  30.0, 4)
    beta_range = np.linspace(0,   0.6, 4)
    fn_range   = np.linspace(0.1, 0.5, 10)

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
    print("phi, beta, fn, cfx, cfy, cmx, cmz")
    for h in hull:
        print("{0:5.2f},{1:5.2f},{2:5.2f},{3:10.2e},{4:10.2e},{5:10.2e},{6:10.2e}".format(h[0],h[1],h[2],h[3],h[4],h[5],h[6]))
    np.save("hull",hull.reshape(len(phi_range),len(beta_range),len(fn_range),7))

    # --- convert ---
    #data = np.zeros(len(phi_range)*len(beta_range)*len(fn_range)*7)
    #data = data.reshape(len(phi_range),len(beta_range),len(fn_range),7)
    #n = 0
    #for i in range(len(phi_range)):
    #    for j in range(len(beta_range)):
    #        for k in range(len(fn_range)):
    #            data[i][j][k] = hull[n]
    #            n += 1
    #np.save("hull",data)

    # --- plot ---
    plot_3d(3,"cx")
    plot_3d(4,"cy")
    plot_3d(5,"ck")
    plot_3d(6,"cn")

    # --- HOW TO USE ---
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
