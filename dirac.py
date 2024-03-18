import numpy as np
import scipy as sp


def dirac_operator_matrix(mesh):
    index_x = np.tile(((mesh.face_he_face[:, 0])*4).reshape(-1,1) + np.arange(4), 4)
    index_y_pre = ((mesh.face_he_face[:, 2])*4).reshape(-1,1) + np.arange(4)
    index_y = np.repeat(index_y_pre, 4, 1)
    Q = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [0,-1,0,0],
        [1,0,0,0],
        [0,0,0,1],
        [0,0,-1,0],
        [0,0,-1,0],
        [0,0,0,-1],
        [1,0,0,0],
        [0,1,0,0],
        [0,0,0,-1],
        [0,0,1,0],
        [0,-1,0,0],
        [1,0,0,0]
    ]).T
    data = (mesh.hyperedges @ Q).flatten()
    i = index_x.flatten()
    j = index_y.flatten()
    in_D = sp.sparse.csr_matrix((data, (i, j)), shape=(4*mesh.num_faces, 4*mesh.num_faces))
    diag_part = sp.sparse.spdiags(np.tile(mesh.face_curvature.reshape(-1,1), (1,4)).flatten(), 0, (4*mesh.num_faces, 4*mesh.num_faces))
    D = in_D - diag_part + 1e-7 * sp.sparse.eye(4*mesh.num_faces)
    return D


def dirac_sv(mesh, dirac_matrix, adj_matrix, eigen_num):
    m1 = 0.5 * adj_matrix.T @ dirac_matrix @ dirac_matrix.T @ adj_matrix
    m2 = sp.sparse.spdiags(np.tile(mesh.vertex_area.reshape(-1,1), (1,4)).flatten(), 0, (4*mesh.num_vertices,4*mesh.num_vertices))
    m1 = 0.5 * (m1 + m1.T)
    m2 = 0.5 * (m2 + m2.T)
    eigen_v, eigen_f = sp.sparse.linalg.eigs(m1, k=eigen_num, M=m2, which='SM', tol=1e-1)
    return eigen_v, eigen_f


def dirac_eigen_alignment(e, w):
    er = np.empty((e.shape[0] // 4, e.shape[1], 4))
    er[:,:,0] = e[::4, :]
    er[:,:,1] = e[1::4, :]
    er[:,:,2] = e[2::4, :]
    er[:,:,3] = e[3::4, :]
    norm_sq = np.sum(er ** 2, axis=2)
    eigen_inv = np.empty_like(er)
    eigen_inv[:,:,0] = er[:, :, 0] / norm_sq
    eigen_inv[:,:,1] = -er[:, :, 1] / norm_sq
    eigen_inv[:,:,2] = -er[:, :, 2] / norm_sq
    eigen_inv[:,:,3] = -er[:, :, 3] / norm_sq
    eigen_inv *= np.tile(w.reshape(-1,1,1), (1,e.shape[1],4))
    scale = np.sum(eigen_inv, axis=0, keepdims=True)
    scale /= np.linalg.norm(scale, axis=2, keepdims=True)
    q = np.tile(scale, (e.shape[0]//4, 1, 1))
    
    pr = er[:,:,0]
    pi = er[:,:,1]
    pj = er[:,:,2]
    pk = er[:,:,3]
    
    qr = q[:,:,0]
    qi = q[:,:,1]
    qj = q[:,:,2]
    qk = q[:,:,3]
    
    res_r = pr*qr-pi*qi-pj*qj-pk*qk
    res_i = pr*qi+pi*qr+pj*qk-pk*qj
    res_j = pr*qj-pi*qk+pj*qr+pk*qi
    res_k = pr*qk+pi*qj-pj*qi+pk*qr
    
    e[0::4, :] = res_r
    e[1::4, :] = res_i
    e[2::4, :] = res_j
    e[3::4, :] = res_k
    return e
