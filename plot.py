from points import *
from LS import *
from scipy.interpolate import UnivariateSpline as interpolator

# Plotting definitions
plt.rcParams["font.family"] = "Lucida Grande"
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

mesh = pymesh.load_mesh('eros_normal_detail.mesh'.format(detail_levels[level]))
P = mesh.nodes; T = mesh.elements
I = trimesh.Trimesh(vertices=P, faces=T, process=False).moment_inertia
Ip, Q = trimesh.inertia.principal_axis(I)
P = matmul(P,Q.T)
T, N, C = surface_normals(P, T)
mesh = pymesh.form_mesh(P,T)

data = np.loadtxt('data-{:s}.dat'.format(detail_levels[level]))
Γ = data.reshape([φ.size, β.size, λ.size, 3])
gammax = np.amax(np.abs(Γ))

F = mesh.elements
I = trimesh.Trimesh(vertices=P, faces=T, process=False).moment_inertia
Ip, Q0 = trimesh.inertia.principal_axis(I)

jjj = 0
for kkk in range(λ.size):
    for iii in range(β.size):
        βi = β[iii]
        λi = λ[kkk]
        R = Rot.from_euler('ZXY', [0, -βi, λi]).as_matrix()
        P2 = matmul(mesh.vertices,R)
        Q = matmul(R, Q0)
        facevectors = np.zeros((F.shape[0],3,3))
        for i, face in enumerate(F):
            for j in range(3):
                facevectors[i][j] = P2[face[j],:]

        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.add_collection3d(mplot3d.art3d.Poly3DCollection(facevectors, facecolor=[0.5,0.5,0.5], lw=0.5,edgecolor=[0,0,0], alpha=0.66))
        ax1.quiver(0,0,0,Q[2,0],Q[2,1],Q[2,2],length=20,normalize=False,color='w', label="body axis")
        ax1.quiver(15,10,-10,0,0,20,length=1,normalize=False,color='r', label="illumination")

        scale = mesh.vertices.flatten()
        ax1.auto_scale_xyz(scale, scale, scale)
        ax1._axis3don = False
        ax1.legend()

        ax2 = fig.add_subplot(1, 2, 2)
        
        x, y = interp(φ, Γ, iii, kkk)
        ax2.plot(x/(2*π),y[:,0],label=r'$Γ_x$')
        ax2.plot(x/(2*π),y[:,1],'--',label=r'$Γ_y$')
        ax2.plot(x/(2*π),y[:,2],':',label=r'$Γ_z$')
        ax2.set_xlabel('Rotational phase')
        #ax2.set_ylabel('Torque')
        ax2.legend(loc=1)
        ax2.set_xlim([0,1])
        ax2.set_ylim([-gammax,gammax])
        plt.suptitle('Torque over rotation about the major principal axis')
        plt.savefig(str(jjj+1)+'.png')
        jjj += 1
        plt.close()
       
