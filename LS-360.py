# -*- coding: utf-8 -*-
import numpy as np, random, matplotlib.pyplot as plt, sys, getopt
import scipy as sc
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from points import *
from LS import *
from scipy.spatial.transform import Rotation as R
from matplotlib import cm

plt.rcParams["font.family"] = "Lucida Grande"
plt.rcParams["font.size"] = 16
plt.style.use("ggplot")

# Silly hacks for avoiding singular points and not showing rotation matrix warnings
zero = 1e-3
import warnings
warnings.filterwarnings("ignore")

def Φ_LS(P, T, C, N, α, φ = 0, β = 0, λ = 0):  
    e_in=np.array([1,0,0])
    R = Rot.from_euler('ZYZ',[λ,-β,φ]).as_matrix().T 
    Rα = Rot.from_euler('z',α).as_matrix()

    e_eye = matmul(Rα,e_in)     
    
    d = 10*np.amax(np.abs(np.linalg.norm(P,axis=1)))*e_in # Displacement
    P = P.dot(R.T) + d
    C = C.dot(R.T) + d
    N = N.dot(R.T)
    vis = (np.dot(e_in,N.T) < 0.)  & (np.dot(e_eye,N.T) < 0.)
    R = np.abs(C.dot(e_in))
    jj = np.argsort(R)

    S = 0.
    for i, j in enumerate(jj):
        if not vis[j]:
            continue
        point = C[j,1:3]
        for jjj in jj[0:i]:
            t = T[jjj]
            if PointInTriangle(point, P[t,1:3]):
                vis[j] = False
        if(vis[j]):
            S += Φ_LS_dA(-N[j], α, e_in, e_eye)
    return S

if __name__=="__main__":
    a = 1
    b = 0.86
    c = 0.82
    n = 250

    P, T = genellip(a, b, c, n, sigma = 0.125, ell = 0.3, gid = 1)  
    T, N, C = surface_normals(P, T)
#    P, T, N, C = ellipsoid_points(a, b, c, n)

    α = np.linspace(0, pi, 33)
    φ = np.linspace(0, 2*pi, 30)
    L = np.zeros([α.size, φ.size])
    
    for i, a in enumerate(α):
        print(i)
        for j, f in enumerate(φ):
           L[i,j] = Φ_LS(P, T, C, N, α=a, φ=f, β=0, λ=0)
    
    X, Y = np.meshgrid(φ*180/pi, α*180/pi)

    fig = plt.figure(figsize=(12,8))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, L, cmap=cm.inferno,
                       linewidth=0, antialiased=False)
    plt.ylabel('α (°)')
    plt.xlabel('φ (°)')
    plt.title(r'$Φ_{LS}(α,φ)$')
    plt.ylim([0,180])
    plt.xlim([0,360])
    plt.legend()
    plt.show()
