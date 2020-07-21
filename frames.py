"""Gaskell Eros Shape Model V1.0 data set, ID:                           
        NEAR-A-MSI-5-EROSSHAPE-V1.0. """
import numpy as np, matplotlib.pyplot as plt, pymesh, trimesh
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from points import *
from LS import *
from scipy.spatial.transform import Rotation as R
from numpy import pi as π, sin, cos, tan, sqrt, log, arctanh
from matplotlib import cm

def fix_mesh(mesh, detail="normal"):
    bbox_min, bbox_max = mesh.bbox;
    diag_len = norm(bbox_max - bbox_min);
    if detail == "normal":
        target_len = diag_len * 2e-2;
    elif detail == "high":
        target_len = diag_len * 10e-3;
    elif detail == "low":
        target_len = diag_len * 4e-2;
    print("Target resolution: {} mm".format(target_len));

    count = 0;
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100);
    mesh, __ = pymesh.split_long_edges(mesh, target_len);
    num_vertices = mesh.num_vertices;
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6);
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len,
                preserve_feature=True);
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100);
        if mesh.num_vertices == num_vertices:
            break;

        num_vertices = mesh.num_vertices;
        print("#v: {}".format(num_vertices));
        count += 1;
        if count > 10: break;

    mesh = pymesh.resolve_self_intersection(mesh);
    mesh, __ = pymesh.remove_duplicated_faces(mesh);
    mesh = pymesh.compute_outer_hull(mesh);
    mesh, __ = pymesh.remove_duplicated_faces(mesh);
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5);
    mesh, __ = pymesh.remove_isolated_vertices(mesh);

    return mesh

P = np.loadtxt('erosP.dat',usecols=(1,2,3))
T = np.loadtxt('erosT.dat',usecols=(1,2,3)).astype(int)-1
mesh = pymesh.form_mesh(P,T)
mesh = fix_mesh(mesh,detail='low')

P = mesh.nodes
T = mesh.elements
I = trimesh.Trimesh(vertices=P,
                       faces=T,
                       process=False).moment_inertia
Ip, Q = trimesh.inertia.principal_axis(I)
P = matmul(P,Q.T)
T, N, C = surface_normals(P, T)

# Use the ZX'Z''- intrinsic rotation notation, where following angles 
# are in corresponding order
#λ = np.array([0.])
#β = np.array([π/3])
λ = np.linspace(0, π, 30)
β = np.linspace(0, 2*π, 31)
φ = np.linspace(0, 2*π, 31)

λn = np.linspace(0, π, 90)
βn = np.linspace(0, 2*π, 91)
φn = np.linspace(0, 2*π, 91)

data = np.loadtxt('data.dat')
Γ = data.reshape([φ.size, β.size, λ.size, 3])

from scipy.interpolate import interpn

gx, gy, gz = np.meshgrid(φ,β,λ)

jjj = 0
for kkk in range(λn.size):
    for iii in range(βn.size):
        βi = βn[iii]
        λi = λn[kkk]
        R = Rot.from_euler('ZXY', [0, -βi, λi]).as_matrix()
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        F = mesh.elements
        I = trimesh.Trimesh(vertices=P,
                           faces=T,
                           process=False).moment_inertia
        Ip, Q = trimesh.inertia.principal_axis(I)

        P2 = matmul(mesh.vertices,R)
        Q = matmul(R, Q)
        facevectors = np.zeros((F.shape[0],3,3))
        for i, face in enumerate(F):
            for j in range(3):
                facevectors[i][j] = P2[face[j],:]

        ax1.add_collection3d(mplot3d.art3d.Poly3DCollection(facevectors, facecolor=[0.5,0.5,0.5], lw=0.5,edgecolor=[0,0,0], alpha=0.66))
        ax1.quiver(0,0,0,Q[2,0],Q[2,1],Q[2,2],length=20,normalize=False,label="body axis")
        ax1.quiver(15,10,-10,0,0,20,length=1,normalize=False,color='k', label="illumination")

        scale = mesh.vertices.flatten()
        ax1.auto_scale_xyz(scale, scale, scale)
        ax1._axis3don = False
        ax1.legend()

        ax2 = fig.add_subplot(1, 2, 2)
        
        gxn, gyn, gzn = np.meshgrid(φn,βi,λi)
        xi = np.vstack(([[gxn.flatten()],[gyn.flatten()],[gzn.flatten()]])).T

        Γ2 = interpn([φ,β,λ], Γ[:,:,:,:], xi)

        ax2.plot(φn/(2*π),Γ2[:,0],label=r'$Γ_x$')
        ax2.plot(φn/(2*π),Γ2[:,1],'--',label=r'$Γ_y$')
        ax2.plot(φn/(2*π),Γ2[:,2],':',label=r'$Γ_z$')
        ax2.set_xlabel('Rotational phase')
        #ax2.set_ylabel('Torque')
        ax2.legend()
        ax2.set_xlim([0,1])
        #ax2.set_ylim([-3,3])
        plt.suptitle('Torque over rotation about the body axis for different body axis orientations')
        plt.savefig(str(jjj+1)+'.png')
        jjj += 1
        
# Rather use ffmpeg for animation, more efficient
#ffmpeg -framerate 60 -i %d.png -c:v libx264 -pix_fmt yuv420p out.mp4
