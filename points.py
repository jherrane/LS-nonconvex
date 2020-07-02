# -*- coding: utf-8 -*-
import numpy as np, random, matplotlib.pyplot as plt, sys, getopt, matplotlib as mpl, h5py, pymesh, seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import uniform as urand, normal, seed
from mpl_toolkits import mplot3d
from numpy import pi, sin, cos, tan, matmul, dot, linspace, arccos, arcsin, sqrt, diag, log, cross, linalg as la
from scipy.special import spherical_in, factorial, lpmn
from matplotlib import rc, rcParams, pyplot as plt
from scipy.spatial import ConvexHull as ch

# Gaussian correlation
def corr(x,ell):
   if(x/ell<10.):
      return np.exp(-0.5*x**2/ell**2)
   return 0.0

# Generate Gaussian random deviates.
def rand_gauss(cv,n,gid=1):
   seed(gid)
   D = np.zeros((n,n))
   
   # Note: cv is assumed to be positive-definite
   eigval, eigvec = la.eigh(cv) 
   
   # In ascending order and drop everything ill-conditioned
   e = eigval[::-1]
   v = np.fliplr(eigvec)
   v[:,e<0.] = 0.0
   e[e<0.] = 0.0
   
   for j in range(n):
      for i in range(n):
         D[i,j] = np.sqrt(cv[i,i]*e[j])*v[i,j]
   
   rn = normal(size=n)
   return np.dot(D,rn)

def deform_surf(h, beta, ell, mesh,gid=1):
   n = mesh.num_vertices
   cv = np.diag(np.ones(mesh.num_vertices))
   
   for i in range(n):
      for j in range(n):
         d = la.norm(mesh.vertices[i,:]-mesh.vertices[j,:])
         cv[i,j] = corr(d, ell)
         cv[j,i] = cv[i,j]
   
   h1 = rand_gauss(cv,n,gid)  
   hn = h*np.exp(beta*h1-0.5*beta**2)
        
   return hn

def sphere_points(n=1000):
   x = np.zeros([3,n])
   offset = 2./n
   increment = pi * (3. - sqrt(5.));

   for i in range(n):
      yi = ((i * offset) - 1) + (offset / 2);
      r = sqrt(1 - yi**2)
      phi = (i % n) * increment
      x[:,i] = [cos(phi) * r, yi, sin(phi) * r]
   return x.T

def surface_fix(V,F):
   n = np.zeros([3,np.shape(F)[0]])
   centroids = np.zeros([3,np.shape(F)[0]])
   for i, face in enumerate(F):
      n[:,i] = cross(V[face[1],:]-V[face[0],:],V[face[2],:]-V[face[0],:])/2.
      centroids[:,i] = (V[face[0],:] + V[face[1],:] + V[face[2],:])/3
      if dot(n[:,i], centroids[:,i]) < 0:
         f0 = face[0]; f1 = face[1]; f2 = face[2]
         F[i,0] = f2
         F[i,1] = f1
         F[i,2] = f0
   return F

def genellip(a=1, b=1, c=1, N = 250, sigma = 0.125, ell = 0.3, gid = 1):
   nodes = sphere_points(n = N)

   elements = ch(nodes).simplices
   elements = surface_fix(nodes, elements)
   M = np.diag([a,b,c])
   nodes = np.dot(M,nodes.T).T
   ellipsoid = pymesh.form_mesh(nodes,elements)
   nodes = deform_mesh(ellipsoid,a, b, c, sigma, ell, gid)
   gellip = pymesh.form_mesh(nodes, ellipsoid.elements)
   tris = gellip.elements
   return nodes, tris

def deform_mesh(mesh, a=1, b=1, c=1, sigma = 0.125, ell = 0.3, gid = 1):
   h     = a*(c/a)**2
   sigma = sigma/h
   beta  = np.sqrt(np.log(sigma**2+1.0))
   # Compute the deformation of surface normals
   hn = deform_surf(h, beta, ell, mesh, gid)
   
   mesh.add_attribute("vertex_normal")
   nn = mesh.get_vertex_attribute("vertex_normal")
   X = np.zeros((mesh.num_vertices,3))
   # Node coordinates:
   for i in range(mesh.num_vertices):
      X[i,:] = mesh.vertices[i,:] + (hn[i]-h)*nn[i,:]
         
   return X

def boundary_faces(T):
   T1 = np.array([T[:,0], T[:,1],T[:,2]]) 
   T2 = np.array([T[:,0], T[:,1],T[:,3]])
   T3 = np.array([T[:,0], T[:,2],T[:,3]]) 
   T4 = np.array([T[:,1], T[:,2],T[:,3]])

   T  = np.concatenate((T1,T2,T3,T4),axis=1)
   T = np.sort(T,axis=0)

   unique_cols, inverse = np.unique(T,axis=1, return_inverse=True)
   counts = np.bincount(inverse)==1
   F = unique_cols[:,counts] 
   
   return F.transpose()
   
"""Collection R, Rphi, Rtheta, J, and p are the ellipsoidal radius parametrization,
its phi- and theta-derivatives, and probability densities."""
def R(a, b, c, theta,phi):
   return a*b*c/sqrt(b**2*c**2*sin(theta)**2*cos(phi)**2 + a**2*c**2*sin(theta)**2*sin(phi)**2 + a**2*b**2*cos(theta)**2)
   
def Rphi(a, b, c, theta,phi):
   return a*b*c**3*sin(theta)**2*(-a**2+b**2)*sin(phi)*cos(phi)/(b**2*c**2*sin(theta)**2*cos(phi)**2 + a**2*c**2*sin(theta)**2*sin(phi)**2 + a**2*b**2*cos(theta)**2)**(3./2)
   
def Rtheta(a, b, c, theta,phi):
   return a*b*c*sin(theta)*cos(theta)*(a**2*b**2 - c**2*(b**2*cos(phi)**2 + a**2*sin(phi)**2))/(b**2*c**2*sin(theta)**2*cos(phi)**2 + a**2*c**2*sin(theta)**2*sin(phi)**2 + a**2*b**2*cos(theta)**2)**(3./2)

def J(a, b, c, theta,phi):
   return R(a, b, c, theta,phi)**2*sin(theta)*sqrt(1 + (Rtheta(a, b, c, theta,phi)/R(a, b, c, theta,phi))**2 + (Rphi(a, b, c, theta,phi)/R(a, b, c, theta,phi)/sin(theta))**2)

def p(a, b, c, theta,phi):
   return 4*np.pi*J(a, b, c, theta,phi)/sin(theta)

def ellipsoid_points_random(a=1, b=1, c=1, n=1000):
   pts = np.zeros((n,3))
   i = 0
   CC = a*b*c*1000
   while i<n:
      theta = arccos(urand(-1,1))
      phi = urand(0,2*np.pi)
      y = urand(0,1)
      r = R(a, b, c, theta,phi)
      pp = p(a, b, c, theta,phi)
      
      if CC*y<=pp:
         pts[i,:] = [r*sin(theta)*np.cos(phi), r*sin(theta)*sin(phi), r*cos(theta)]
         i += 1
#         print('%.f/%.f'%(i,n))
   
   x = np.column_stack(pts)
   tris = ch(x.T).simplices
   tris, normals, centroids = surface_normals(x.T,tris)
   return x.T, tris, normals, centroids

def hard_sphere(a=1, b=1, c=1, n=1002):
   """Hard coded generator of sphere by the octant. Originally contains 400
   nodes, but the usage of convex hull triangulation allows variable node
   numbers. I think..."""
   Nnod = n
   Ntri = (n-2)*2
   Ntr = int(sqrt(Ntri/8))
   mu = np.zeros(Nnod)
   phi = np.zeros(Nnod)
   u = np.zeros((Nnod,3))
   i = np.zeros((2*Nnod,3))
   Njj = np.zeros((361,721))
   
   nnod = 0
   u[nnod,:] = [0.,0.,1.]
   mu[nnod] = 1.
   phi[nnod] = 0.
   Njj[0,0] = nnod
   
   for j1 in range(Ntr):
      the = j1*pi/(2*Ntr)
      ct = cos(the)
      st = sin(the)
      
      for j2 in range(4*j1):
         phi = j2*pi/(2*j1)
         cf = cos(phi)
         sf = sin(phi)
         
         nnod += 1
         u[nnod, :] = [st*cf, st*sf, ct]
         mu[nnod] = ct
         phi[nnod] = phi
         Njj[j1,j2] = nnod
         if(j2==0): Njj[j1,4*j1]=nnod
         
   for j1 in range(Ntr, 0, -1):
      the = (2*Ntr-j1)*pi/(2*Ntr)
      ct = cos(the)
      st = sin(the)
      
      for j2 in range(4*j1):
         phi = j2*pi/(2*j1)
         cf = cos(phi)
         sf = sin(phi)
         
         nnod += 1
         u[nnod,:] = [st*cf, st*sf, ct]
         mu[nnod] = ct
         phi[nnod] = phi
         Njj[2*Ntr-j1,j2] = nnod
         if(j2==0): Njj[2*Ntr-j1,4*j1] = nnod
         
   nnod +=1
   u[nnod,:] = [0.,0.,-1.]
   mu[nnod] = -1.
   phi[nnod] = 0.
   Njj[2*Ntr,0] = nnod
   
   if nnod != 4*Ntr**2+1:
      print('Error: number of nodes inconsistent.')
      print(nnod, 4*Ntr**2+1)
      sys.exit(1)
   
   u[:,0] = a*u[:,0]
   u[:,1] = b*u[:,1]
   u[:,2] = c*u[:,2]
   
   tris = ch(u).simplices
   tris, normals, centroids = surface_normals(u,tris)
   return u, tris, normals, centroids
   
def ellipsoid_points(a=1,b=1,c=1,n=1000):
   """
   Grid of points on an ellipsoid. Generated using a so called
   Fibonacci sphere algorithm, scaled to fill an ellipsoidal surface.
   Because of this scaling, the method is not guaranteed to be optimal.
   However, evenly parametrizing points on an ellipsoid is analytically
   too much work for a single night with a python script...
   """ 
   x = np.zeros([3,n])
   offset = 2./n
   increment = pi * (3. - sqrt(5.));

   for i in range(n):
      yi = ((i * offset) - 1) + (offset / 2);
      r = sqrt(1 - yi**2)
      phi = (i % n) * increment
      
      x[:,i] = [a*cos(phi) * r, b*yi, c*sin(phi) * r]
   
   tris = ch(x.T).simplices
   tris, normals, centroids = surface_normals(x.T,tris)
   return x.T, tris, normals, centroids

def mesh_vectors(V,F):
   """
   Creates a vector set of the mesh data for nice plotting.
   """
   msh = np.zeros((np.shape(F)[0],3,3))
   for i, face in enumerate(F):
      for j in range(3):
         msh[i][j] = V[face[j],:]
   return msh
   
def surface_normals(V,F):
   """
   Surface normal is simply the cross product of vectors CB and CA, in
   the said order.
   
   The centroid is given by (1/3)*(a+b+c), where a, b, and c are the
   triangle vertex vectors.
   """
   FF = np.zeros(F.shape)
   n = np.zeros([np.shape(F)[0],3])
   centroids = np.zeros([np.shape(F)[0],3])
   for i, face in enumerate(F):
      n[i,:] = cross(V[face[1],:]-V[face[0],:],V[face[2],:]-V[face[0],:])/2.
      centroids[i,:] = (V[face[0],:] + V[face[1],:] + V[face[2],:])/3
      if dot(n[i,:], centroids[i,:]) < 0:
         FF[i,0] = face[2]
         FF[i,1] = face[1]
         FF[i,2] = face[0]
         n[i,:] = -n[i,:]
      else: FF[i,:] = face
   return F, n, centroids

def PointInTriangle(pt,tri):
    '''checks if point pt(2) is inside triangle tri(3x2).'''
    a = 1/(-tri[1,1]*tri[2,0]+tri[0,1]*(-tri[1,0]+tri[2,0])+ \
        tri[0,0]*(tri[1,1]-tri[2,1])+tri[1,0]*tri[2,1])
    s = a*(tri[2,0]*tri[0,1]-tri[0,0]*tri[2,1]+(tri[2,1]-tri[0,1])*pt[0]+ \
        (tri[0,0]-tri[2,0])*pt[1])
    if s<0: return False
    else: t = a*(tri[0,0]*tri[1,1]-tri[1,0]*tri[0,1]+(tri[0,1]-tri[1,1])*pt[0]+ \
              (tri[1,0]-tri[0,0])*pt[1])
    return ((t>0) and (1-s-t>0))
    
def plane_projection(p, normal):
   return p-np.dot(p,normal)*normal/np.linalg.norm(normal)**2

def mesh_analysis(pts, tris, normals):
   print('Number of triangles N = %.f'%(np.shape(normals)[0]))
   norms = np.linalg.norm(normals,axis=1)
   mean = np.mean(norms)
   std = np.std(norms)
   print('Mean triangle area is mu = %.3e, with σ = %.3e (%.1f %%)'%(mean,std, std/mean*100))
   print('Area of the surface = %.3f, 4π = %.3f'%(np.sum(norms),4*np.pi))
   fig = plt.figure(figsize=(8,8))
   sns.distplot(norms, kde=False, color="b", hist=True, bins=200, norm_hist=False)
   plt.title('Distribution of triangle areas')
   plt.ylabel('count')
   plt.xlabel('area')

def plot_mesh(points, tris, centroids, normals, quivers=False):
   fig = plt.figure(figsize=(8,8))
   ax = mplot3d.Axes3D(fig)
 
   if quivers: 
      x,y,z = centroids
      u,v,w = normals.T
      ax.quiver(x,y,z,u,v,w,length=0.2,normalize=True)
   
   meshvectors = mesh_vectors(points, tris)
   ax.add_collection3d(mplot3d.art3d.Poly3DCollection(meshvectors, facecolor=[0.5,0.5,0.5], lw=0.5, edgecolor=[0,0,0], alpha=.8, antialiaseds=True))  
   scale = points.flatten('F')
   ax.auto_scale_xyz(scale, scale, scale) 
   
   ax.set_xlabel('x')
   ax.set_ylabel('y')
   ax.set_zlabel('z')
   plt.show()
   return fig

