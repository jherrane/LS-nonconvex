import numpy as np, matplotlib.pyplot as plt, pymesh, scipy as sc
from mpl_toolkits import mplot3d
from numpy import pi, sin, cos, tan, matmul, dot, linspace, arccos, arcsin, sqrt, diag, log, cross, dot, arctanh
from numpy.linalg import norm
from points import *
from scipy.spatial.transform import Rotation as Rot

plt.rcParams['font.family'] = 'Lucida Grande'
plt.rcParams['font.size'] = 16
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')

def Φ_LS_a(a, b, c, α, φ=0, β=0, λ=0):
    """Determines the integral phase function Φ(α) = I(α)/I(0).
    We work in the xy-plane, so phase angle can be used to construct a unambiguous
    rotation matrix in order to find e_out. Angle gamma is used to rotate the
    scattering body about z''-axis
    """
    e_in = np.array([1, 0, 0])
    if pi - α < 1e-07: return 0.0

    Rα = Rot.from_euler('z', α).as_matrix()
    R = Rot.from_euler('ZYZ', [λ, -β, φ]).as_matrix()
    e_out = matmul(Rα, e_in)
    e_out = matmul(R.T, e_out)
    e_in = matmul(R.T, e_in)
    
    C = diag([1.0 / a ** 2, 1.0 / b ** 2, 1.0 / c ** 2])
    S_in = sqrt(dot(e_in, matmul(C, e_in)))
    S_out = sqrt(dot(e_out, matmul(C, e_out)))
    if dot(e_in, matmul(C, e_out)) / (S_in * S_out) > 1.0:
        αp = arccos(1.0)
    else:
        αp = arccos(dot(e_in, matmul(C, e_out)) / (S_in * S_out))
   
    S = sqrt(S_in ** 2 + S_out ** 2 + 2 * S_in * S_out * cos(αp))
    λp = arccos((S_in + S_out * cos(αp)) / S)
    
    if αp - λp < 1e-07:
        return a * b * c * S_in * S_out / S * (cos(λp - αp) + cos(λp))
    
    Φ = a * b * c * S_in * S_out / S * (cos(λp - αp) + cos(λp) - sin(λp) * sin(λp - αp) * log(tan(λp / 2) * tan((αp - λp) / 2)))
    return Φ

def Φ_LS_dA(n, alpha, e_in, e_obs):
    """Determine the phase function contribution of a single area element using
    the Lommel-Seeliger reflectance function 1/(μ-μ0). The normal vector length
    gives the area of the element."""
    dA = norm(n)
    nhat = n / dA
    mu0 = dot(nhat, e_in)
    mu = dot(nhat, e_obs)
    if mu < 0 or mu0 < 0:
        return 0.0
    else:
        R_LS = 1.0 / (mu0 + mu)
        return dA * mu * mu0 * R_LS * 2 / pi

def Φ_LS_n(N, α, φ=0, β=0, λ=0):
    """Calculates the Lommel-Seeliger integral phase function numerically."""
    e_in = np.array([1, 0, 0])
    Rα = Rot.from_euler('z', α).as_matrix()
    R = Rot.from_euler('ZYZ', [φ, -β, λ]).as_matrix()
    e_out = matmul(Rα, e_in)
    S = 0.0
    for i, n in enumerate(N):
        S += Φ_LS_dA(n, α, e_in, e_out)
    return S

def Φ_LS(P, T, C, N, α, φ=0, β=0, λ=0):
    e_in = np.array([1, 0, 0])
    Rα = Rot.from_euler('z', α).as_matrix()
    R = Rot.from_euler('ZYZ', [λ, -β, φ]).as_matrix()
    e_eye = matmul(Rα, e_in)
    d = 10 * np.amax(np.abs(np.linalg.norm(P, axis=1))) * e_in
    P = P.dot(R.T) + d
    C = C.dot(R.T) + d
    N = N.dot(R.T)
    
    vis = (np.dot(e_in, N.T) < 0.0) & (np.dot(e_eye, N.T) < 0.0)
    R = np.abs(C.dot(e_in))
    jj = np.argsort(R)
    
    S = 0.0
    
    for i, j in enumerate(jj):
        if not vis[j]: continue
        point = C[j, 1:3]
        for jjj in jj[0:i]:
            t = T[jjj]
            if PointInTriangle(point, P[t, 1:3]):
                vis[j] = False
        if vis[j]:
            S += Φ_LS_dA(-N[j], α, e_in, e_eye)
    return S

def Γ_dA(r, n, e_in):
    dA = norm(n)
    nhat = n / dA
    mu0 = dot(nhat, e_in)
    dN = 2 * pi * mu0 * (mu0 * log(mu0 / (mu0 + 1)) + 1) * cross(nhat, r)
    return dN * dA

def Γ_LS(P, T, C, N, φ=0, β=0, λ=0, convex=True):
    """Calculates the Lommel-Seeliger integral phase function numerically."""
    e_in = np.array([1, 0, 0])
    R = Rot.from_euler('ZYZ', [φ, -β, λ]).as_matrix()
    
    d = 10 * np.amax(np.abs(np.linalg.norm(P, axis=1))) * e_in
    P = P.dot(R) + d
    C = C.dot(R)
    D = C + d
    N = N.dot(R)
    
    vis = np.dot(e_in, N.T) > 0.0
    Q = np.abs(D.dot(e_in))
    jj = np.argsort(Q)
    Γ = np.array([0.0, 0.0, 0.0])
    
    if convex:
        for i, n in enumerate(N):
            if not vis[i]: continue
            Γ += Γ_dA(C[i, :], n, e_in)
    else:
        for i, j in enumerate(jj):
            if not vis[j]: continue
            else:
                point = D[j, 1:3]
                for jjj in jj[0:i]:
                    t = T[jjj]
                    if PointInTriangle(point, P[t, 1:3]):
                        vis[j] = False
                if vis[j]:
                    Γ += Γ_dA(C[j, :], N[j], e_in)
    return Γ

if __name__ == '__main__':
    a = 1
    b = 0.86
    c = 0.82
    n = 250
    gell = True

    if gell == True:
        P, T = genellip(a, b, c, (n + 2), sigma=0.12, ell=0.3)
        T, N, C = surface_normals(P, T)
    else:
        P, T, N, C = ellipsoid_points(a, b, c, n)

    α = np.linspace(0, pi, 33)
    Φ = np.zeros(α.size)
    Φn = np.zeros(α.size)
    for i, αi in enumerate(α):
        Φ[i] = Φ_LS_a(a, b, c, α=αi, λ=λ)
        Φn[i] = Φ_LS_n(N, α=αi, λ=λ)

    if gell:
        for i, αi in enumerate(α):
            Φn[i] = Φ_LS(P, T, C, N, α=αi, λ=λ)

    fig = plt.figure(figsize=(12, 8))
    plt.plot((α * 180 / pi), Φ, label=('Analytic with (a,b,c)=(%3.2f,%3.2f,%3.2f)' % (a, b, c)))
    if gell:
        plt.plot((α * 180 / pi), Φn, '--', label='Numerical solution with correction')
    else:
        plt.plot((α * 180 / pi), Φn, '--', label='Numerical solution without correction')
    plt.title('Integral phase function with %.f triangles' % np.shape(N)[0])
    plt.xlabel('α (°)')
    plt.ylabel('Φ(α)')
    plt.xlim([0, 180])
    plt.legend()
    plt.show()
