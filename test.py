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

def draw_mesh(mesh):
    fig = plt.figure(figsize=(6, 6),frameon=False)
    ax = mplot3d.Axes3D(fig)

    # Collect face data as vectors for plotting
    F = mesh.elements

    facevectors = np.zeros((F.shape[0],3,3))
    for i, face in enumerate(F):
        for j in range(3):
            facevectors[i][j] = mesh.vertices[face[j],:]
    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(facevectors, facecolor=[0.5,0.5,0.5], lw=0.5,edgecolor=[0,0,0], alpha=0.66))
    
    scale = mesh.vertices.flatten()
    
    I = trimesh.Trimesh(vertices=P,
                       faces=T,
                       process=False).moment_inertia
    Ip, Q = trimesh.inertia.principal_axis(I)
    ax.quiver(0,0,0,Q[2,0],Q[2,1],Q[2,2],length=20,normalize=False)
    ax.quiver(-10,-15,0,1.3,0,0,length=20,normalize=False,color='k')
    ax.auto_scale_xyz(scale, scale, scale)

    plt.show()
    return fig

mesh = pymesh.form_mesh(P,T)
fig = draw_mesh(mesh)

# Use the ZX'Z''- intrinsic rotation notation, where following angles 
# are in corresponding order
#λ = np.array([0.])
#β = np.array([π/3])
λ = np.linspace(0, 2*π, 50)
β = np.linspace(0, π, 50)
φ = np.linspace(0, 2*π, 50)

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

data = np.loadtxt('data.dat')
#data = np.loadtxt('data_nonconvex.dat')
#data0 = np.loadtxt('data_convex.dat')
Γ = data.reshape([φ.size, β.size, λ.size, 3])
#Γ0 = data0.reshape([φ.size, β.size, λ.size, 3])
#Γ2 = (Γ-Γ0)/Γ

Γ2 = Γ
jjj = 0

import imageio
image_list = []
for i in range(1,4951):
    image_list.append(imageio.imread(str(i)+'.png'))
imageio.mimwrite('anim_long.mp4', image_list, fps=30)
