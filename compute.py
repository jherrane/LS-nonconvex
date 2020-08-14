"""
Here are all definitions, this must be initiated before any other cell can be run.

Gaskell Eros Shape Model V1.0 data set, ID: NEAR-A-MSI-5-EROSSHAPE-V1.0. 
"""
from points import *
from LS import *
from scipy.interpolate import UnivariateSpline as interpolator

# Plotting definitions
plt.rcParams['font.family'] = 'Fira Sans Medium'
plt.rcParams['font.size'] = 16
plt.style.use('dark_background')

# Function definitions
def interp(φ,Γ,ib,il):
    φn = np.linspace(0, 2*π, 200)
    vals = Γ[:, ib, il, :]
    Γip = np.zeros([φn.size, 3])

    for i in range(3):
        spl = interpolator(φ, vals[:,i])
        spl.set_smoothing_factor(0.5)
        Γip[:,i] = spl(φn)
    return φn, Γip

# Calculations are done as follows: Probe the angles determining the direction of the 
# body axis (in red). Latitude β measured as the angle between body axis and 
# illumination (in black), given by positive rotation of β in (0,π) about the x-axis. 
# Angle λ in (0,2π) corresponds to rotation about the y-axis. All in all we
# use the YX'Z''- intrinsic rotation notation, where initially rotate λ about z-axis,
# then β about the new x-axis and finally rotate the body about the final z-axis

λ = np.linspace(0, π, 50)
β = np.linspace(0, 2*π, 50)
φ = np.linspace(0, 2*π, 50)

"""
Here we choose which level of detail (don't worry, everything is precalculated) to use. 
This way, it is easier to compare how convergence should work here.
"""

detail_levels = ['low', 'normal', 'high']

level = 2

mesh = pymesh.load_mesh('eros_{:s}_detail.mesh'.format(detail_levels[level]))
P = mesh.nodes; T = mesh.elements
I = trimesh.Trimesh(vertices=P, faces=T, process=False).moment_inertia
Ip, Q = trimesh.inertia.principal_axis(I)
P = matmul(P,Q.T)
T, N, C = surface_normals(P, T)
mesh = pymesh.form_mesh(P,T)

import time, datetime
Γ = np.zeros([φ.size, β.size, λ.size, 3])

start = time.time()
for i, fi in enumerate(φ):
    print((i+1),'/',φ.size)
    for j, bj in enumerate(β):
        for k, lk in enumerate(λ):
            Γ[i,j,k:] = Γ_LS(P, T, C, N, φ = fi, β = bj, λ = lk, convex = True)
end = time.time()
print('Done in', str(datetime.timedelta(seconds=np.round(end-start,0))))

