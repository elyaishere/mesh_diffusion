import numpy as np
import scipy as sp
from mesh import Mesh
from dirac import *
from quaternions import spin_transform

def wilmore_flow(V, F, step): # step < 0.5
    mesh = Mesh(V, F)
    
    dirac_matrix = dirac_operator_matrix(mesh)

    rho = step * mesh.face_curvature
    rho -= np.sum(rho) / np.sum(mesh.face_area) * mesh.face_area
    dirac_matrix += sp.sparse.spdiags(np.tile(rho.reshape(-1,1), (1,4)).flatten(), 0, (4*mesh.num_faces, 4*mesh.num_faces))

    adj_matrix = mesh.vf_adjacent_matrix()

    _, eigen_f = dirac_sv(mesh, dirac_matrix, adj_matrix, 1)
    eigen_f = dirac_eigen_alignment(eigen_f, mesh.vertex_area())

    V_new = spin_transform(mesh, adj_matrix @ eigen_f)
    return V_new
