import igl
import numpy as np
import scipy as sp


class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        self.num_vertices = len(vertices)
        self.num_faces = len(faces)

        neibour_face, _ = igl.triangle_triangle_adjacency(faces)
        neibour_face[:, :2] = neibour_face[:, 1::-1]
        neibour_face[:, 1:] = neibour_face[:, :0:-1]

        #list_num_boundaries = np.sum(neibour_face == -1, axis=1)
        neibour_face = neibour_face.flatten('F')

        face_he_face = np.column_stack((np.tile(range(faces.shape[0]), 3), range(neibour_face.size), neibour_face))
        self.face_he_face = face_he_face[neibour_face != -1, :]

        self.halfedges = igl.oriented_facets(faces)
        self.halfedges = self.halfedges[self.face_he_face[:,1],:]
        
        self.halfedges_coordinate = self._get_halfedges_coordinate()
        self.meancurvature_edge = self._get_meancurvature_edge()
        self.hyperedges = self._get_hyperedges()
        self.face_curvature = self._get_face_curvature()
        self.face_area = self._get_face_area()

        rows = np.arange(self.num_faces)*4
        cols = np.array(self.faces)*4
        rows = np.tile(rows, (4,1)) + np.arange(4).reshape((4,1))
        rows = rows.reshape((-1, 1), order='F')
        rows = np.tile(rows, (1,3)).reshape((-1, 1), order='F')
        rows = rows.flatten()
        cols = cols.reshape((-1, 1), order='F')
        cols = np.tile(cols, (1, 4)) + np.arange(4).reshape((1,4))
        cols = cols.flatten()
        self.vf_adjacent_matrix = sp.sparse.csr_matrix((np.ones_like(rows, dtype=float), (rows, cols)), shape=(4*self.num_faces, 4*self.num_vertices))

        rows = self.faces.flatten('F')
        cols = np.tile(np.arange(self.num_faces), 3).flatten('F')
        adj = sp.sparse.csr_matrix((np.ones_like(rows, dtype=float), (rows, cols)), (self.num_vertices, self.num_faces))
        self.vertex_area = adj @ self.face_area / 3
    
    def _get_halfedges_coordinate(self):
        halfedges_coordinate = -self.vertices[self.faces[:, [1, 2, 0]].flatten('F'),:] + self.vertices[self.faces[:,[2, 0, 1]].flatten('F'), :]
        halfedges_coordinate = halfedges_coordinate[self.face_he_face[:, 1], :]
        return halfedges_coordinate
    
    def _get_meancurvature_edge(self):
        edge_norm_square = np.sum(self.halfedges_coordinate**2, axis=1, keepdims=True)
        normal_face = igl.per_face_normals(self.vertices, self.faces, np.array([1/3, 1/3, 1/3])**0.5)
        normals_right = normal_face[self.face_he_face[:, 2], :]
        normal_face_extend = normal_face[self.face_he_face[:, 0], :]
        A = normal_face_extend - normals_right - 2 * self.halfedges_coordinate / edge_norm_square
        meancurvature_edge = np.sum(normal_face_extend * A, axis=1, keepdims=True)
        A = np.cross(normal_face_extend, normals_right)
        B = self.halfedges_coordinate / edge_norm_square
        C = np.sum(A*B, axis=1, keepdims=True)
        meancurvature_edge = meancurvature_edge / C
        plane_index = np.abs(C) < (1e-5) * np.min(edge_norm_square)
        meancurvature_edge[plane_index] = 0
        return meancurvature_edge
    
    def _get_hyperedges(self):
        return np.column_stack((self.meancurvature_edge, self.halfedges_coordinate))
    
    def _get_face_curvature(self):
        face_curvature = np.zeros((3 * self.num_faces, 1))
        face_curvature[self.face_he_face[:, 1], :] = self.meancurvature_edge
        return np.sum(np.reshape(face_curvature, (-1, 3), order='F'), axis=1)
    
    def _get_face_area(self):
        return igl.doublearea(self.vertices, self.faces) / 2
    
    @property
    def willmore_energy(self):
        return np.sum(self.face_curvature**2 / self.face_area ) / 4
    
    def update_vertices(self, vertices):
        self.vertices = vertices
        assert self.num_vertices == len(vertices)
        self.halfedges_coordinate = self._get_halfedges_coordinate()
        self.meancurvature_edge = self._get_meancurvature_edge()
        self.hyperedges = self._get_hyperedges()
        self.face_curvature = self._get_face_curvature()
        self.face_area = self._get_face_area()
