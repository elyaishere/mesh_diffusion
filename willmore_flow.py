import numpy as np
import scipy as sp
from time import time
from mesh import Mesh
from dirac import *
from quaternions import spin_transform

def willmore_flow(mesh: Mesh, step: float): # step < 0.5
    t = time()
    dirac_matrix = dirac_operator_matrix(mesh)
    print(f"dirac matrix construction took {time() - t} sec.")

    rho = step * mesh.face_curvature
    rho -= np.sum(rho) / np.sum(mesh.face_area) * mesh.face_area
    dirac_matrix += sp.sparse.spdiags(np.tile(rho.reshape(-1,1), (1,4)).flatten(), 0, (4*mesh.num_faces, 4*mesh.num_faces))

    t = time()
    _, eigen_f = dirac_sv(mesh, dirac_matrix, mesh.vf_adjacent_matrix, 1)
    print(f"finding singular values took {time() - t} sec.")
    eigen_f = dirac_eigen_alignment(eigen_f, mesh.vertex_area)

    t = time()
    V_new = spin_transform(mesh, mesh.vf_adjacent_matrix @ eigen_f)
    print(f"soin transform took {time() - t} sec.")
    return V_new
