import numpy as np
import scipy as sp
from ismember import ismember


def quat_conj(q):
    if q.shape[2] == 4:
        qr=q[:,:,0]
        qi=q[:,:,1]
        qj=q[:,:,2]
        qk=q[:,:,3]
        return np.stack([qr,-qi,-qj,-qk], 2)
    qr=q[::4,:]
    qi=q[1::4,:]
    qj=q[2::4,:]
    qk=q[3::4,:]
    res = np.zeros((q.shape[0], q.shape[1]))
    res[::4,:] = qr
    res[1::4,:] = -qi
    res[2::4,:] = -qj
    res[3::4,:] = -qk
    return res


def quat_prod(p,q):
    if p.shape[2] == 4:
        pr=p[:,:,0]
        pi=p[:,:,1]
        pj=p[:,:,2]
        pk=p[:,:,3]
        qr=q[:,:,0]
        qi=q[:,:,1]
        qj=q[:,:,2]
        qk=q[:,:,3]
        res_r=pr*qr-pi*qi-pj*qj-pk*qk
        res_i=pr*qi+pi*qr+pj*qk-pk*qj
        res_j=pr*qj-pi*qk+pj*qr+pk*qi
        res_k=pr*qk+pi*qj-pj*qi+pk*qr
        return np.stack([res_r,res_i,res_j,res_k], 2)
    pr=p[::4,:]
    pi=p[1::4,:]
    pj=p[2::4,:]
    pk=p[3::4,:]
    qr=q[::4,:]
    qi=q[1::4,:]
    qj=q[2::4,:]
    qk=q[3::4,:]
    res_r=pr*qr-pi*qi-pj*qj-pk*qk
    res_i=pr*qi+pi*qr+pj*qk-pk*qj
    res_j=pr*qj-pi*qk+pj*qr+pk*qi
    res_k=pr*qk+pi*qj-pj*qi+pk*qr
    res = np.zeros((p.shape[0], p.shape[1]))
    res[::4,:] = res_r
    res[1::4,:] = res_i
    res[2::4,:] = res_j
    res[3::4,:] = res_k
    return res


def spin_transform(mesh, phi):
    phi = np.real(phi)
    phi /= np.linalg.norm(phi)
    phi *= (mesh.num_faces ** 0.5)
    phi = np.stack([phi[::4], phi[1::4], phi[2::4], phi[3::4]], 2)
    hyperedges = np.stack([mesh.hyperedges[:, :1], mesh.hyperedges[:, 1:2], mesh.hyperedges[:, 2:3], mesh.hyperedges[:, 3:4]], 2)
    face_id_indices = mesh.face_he_face[:, 0]
    f_edge = quat_prod(quat_prod(quat_conj(phi[face_id_indices, : , :]),hyperedges), phi[mesh.face_he_face[:, 2], :, :])
    edges = np.unique(np.sort(mesh.halfedges, axis=1), axis=0)
    _, edges_halfedges = ismember(edges, mesh.halfedges, 'rows')

    index_x = range(edges.shape[0])
    index_y_minus = edges[:, 0]
    index_y_plus = edges[:, 1]
    
    bx = f_edge[edges_halfedges,:,1]
    by = f_edge[edges_halfedges,:,2]
    bz = f_edge[edges_halfedges,:,3]

    A = sp.sparse.csr_matrix((-np.ones_like(index_x, dtype=float), (index_x, index_y_minus)), shape=(edges.shape[0], mesh.num_vertices))
    B = sp.sparse.csr_matrix((np.ones_like(index_x, dtype=float), (index_x, index_y_plus)), shape=(edges.shape[0], mesh.num_vertices))
    matrix = A + B
    matrix = matrix[:, :-1]

    x = sp.sparse.linalg.spsolve(matrix.T @ matrix, matrix.T @ bx)
    y = sp.sparse.linalg.spsolve(matrix.T @ matrix, matrix.T @ by)
    z = sp.sparse.linalg.spsolve(matrix.T @ matrix, matrix.T @ bz)
    f_v = np.stack([x,y,z],1)
    f_v = np.vstack([f_v, np.zeros((1,3))])
    return f_v
