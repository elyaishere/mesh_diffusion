from pathlib import Path
from pytorch3d.ops.knn import knn_points
import torch
import numpy as np
import igl
import copy
import pickle


class Mesh:

    def __init__(self, file, hold_history=False, vs=None, faces=None, device='cpu', gfmm=True):
        if file is None:
            return
        self.filename = Path(file)
        self.vs = self.v_mask = self.edge_areas = None
        self.edges = self.gemm_edges = self.sides = None
        self.device = device
        if vs is not None and faces is not None:
            self.vs, self.faces = vs.cpu().numpy(), faces.cpu().numpy()
            self.scale, self.translations = 1.0, np.zeros(3,)
        else:
            self.vs, self.faces = igl.read_triangle_mesh(file)
            self.normalize_unit_bb()
        self.vs_in = copy.deepcopy(self.vs)
        self.v_mask = np.ones(len(self.vs), dtype=bool)
        self.build_gemm()
        self.history_data = None
        if hold_history:
            self.init_history()
        if gfmm:
            self.gfmm = self.build_gfmm() #TODO get rid of this DS
        else:
            self.gfmm = None
        if type(self.vs) is np.ndarray:
            self.vs = torch.from_numpy(self.vs)
        if type(self.faces) is np.ndarray:
            self.faces = torch.from_numpy(self.faces)
        self.vs = self.vs.to(self.device)
        self.faces = self.faces.to(self.device).long()
        self.area, self.normals = self.face_areas_normals(self.vs, self.faces)

    def build_gemm(self):
        self.ve = [[] for _ in self.vs]
        self.vei = [[] for _ in self.vs]
        edge_nb = []
        sides = []
        edge2key = dict()
        edges = []
        edges_count = 0
        nb_count = []
        for face in self.faces:
            faces_edges = [tuple(sorted((face[i], face[(i + 1) % 3]))) for i in range(3)]
            for idx, edge in enumerate(faces_edges):
                edge = tuple(sorted(list(edge)))
                faces_edges[idx] = edge
                if edge not in edge2key:
                    edge2key[edge] = edges_count
                    edges.append(list(edge))
                    edge_nb.append([-1, -1, -1, -1])
                    sides.append([-1, -1, -1, -1])
                    self.ve[edge[0]].append(edges_count)
                    self.ve[edge[1]].append(edges_count)
                    self.vei[edge[0]].append(0)
                    self.vei[edge[1]].append(1)
                    nb_count.append(0)
                    edges_count += 1
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]
                edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]
                nb_count[edge_key] += 2
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
                sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2
        self.edges = np.array(edges, dtype=np.int32)
        self.gemm_edges = np.array(edge_nb, dtype=np.int64)
        self.sides = np.array(sides, dtype=np.int64)
        self.edges_count = edges_count
        # lots of DS for loss
        self.nvs, self.nvsi, self.nvsin = [], [], []
        for i, e in enumerate(self.ve):
            self.nvs.append(len(e))
            self.nvsi.append(len(e) * [i])
            self.nvsin.append(list(range(len(e))))
        self.vei = torch.from_numpy(np.concatenate(self.vei).ravel()).to(self.device).long()
        self.nvsi = torch.from_numpy(np.concatenate(self.nvsi).ravel()).to(self.device).long()
        self.nvsin = torch.from_numpy(np.concatenate(self.nvsin).ravel()).to(self.device).long()
        ve_in = copy.deepcopy(self.ve)
        self.ve_in = torch.from_numpy(np.concatenate(ve_in).ravel()).to(self.device).long()
        self.max_nvs = max(self.nvs)
        self.nvs = torch.Tensor(self.nvs).to(self.device).float()
        self.edge2key = edge2key

    def build_ef(self):
        edge_faces = dict()
        if type(self.faces) == torch.Tensor:
            faces = self.faces.cpu().numpy()
        else:
            faces = self.faces
        for face_id, face in enumerate(faces):
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                if edge not in edge_faces:
                    edge_faces[edge] = []
                edge_faces[edge].append(face_id)
        for k in edge_faces.keys():
            if len(edge_faces[k]) < 2:
                edge_faces[k].append(edge_faces[k][0])
        return edge_faces

    def build_gfmm(self):
        edge_faces = self.build_ef()
        gfmm = []
        if type(self.faces) == torch.Tensor:
            faces = self.faces.cpu().numpy()
        else:
            faces = self.faces
        for face_id, face in enumerate(faces):
            neighbors = [face_id]
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                neighbors.extend(list(set(edge_faces[edge]) - set([face_id])))
            gfmm.append(neighbors)
        return torch.Tensor(gfmm).long().to(self.device)

    def normalize_unit_bb(self):
        """
        normalizes to unit bounding box and translates to center
        if no
        :param verts: new verts
        """
        cache_norm_file = self.filename.with_suffix('.npz')
        if not cache_norm_file.exists():
            scale = max([self.vs[:, i].max() - self.vs[:, i].min() for i in range(3)])
            scaled_vs = self.vs / scale
            target_mins = [(scaled_vs[:, i].max() - scaled_vs[:, i].min()) / -2.0 for i in range(3)]
            translations = [(target_mins[i] - scaled_vs[:, i].min()) for i in range(3)]
            np.savez_compressed(cache_norm_file, scale=scale, translations=translations)
        # load from the cache
        cached_data = np.load(cache_norm_file, encoding='latin1', allow_pickle=True)
        self.scale, self.translations = cached_data['scale'], cached_data['translations']
        self.vs /= self.scale
        self.vs += self.translations[None, :]

    def discrete_project(self, pc: torch.Tensor, thres=0.9, cpu=False):
        with torch.no_grad():
            device = torch.device('cpu') if cpu else self.device
            pc = pc.double()
            if isinstance(self, Mesh):
                mid_points = self.vs[self.faces].mean(dim=1)
                normals = self.normals
            else:
                mid_points = self[:, :3]
                normals = self[:, 3:]
            pk12 = knn_points(mid_points[:, :3].unsqueeze(0), pc[:, :, :3], K=3).idx[0]
            pk21 = knn_points(pc[:, :, :3], mid_points[:, :3].unsqueeze(0), K=3).idx[0]
            loop = pk21[pk12].view(pk12.shape[0], -1)
            knn_mask = (loop == torch.arange(0, pk12.shape[0], device=self.device)[:, None]).sum(dim=1) > 0
            mid_points = mid_points.to(device)
            pc = pc[0].to(device)
            normals = normals.to(device)[~ knn_mask, :]
            masked_mid_points = mid_points[~ knn_mask, :]
            displacement = masked_mid_points[:, None, :] - pc[:, :3]
            torch.cuda.empty_cache()
            distance = displacement.norm(dim=-1)
            mask = (torch.abs(torch.sum((displacement / distance[:, :, None]) *
                                        normals[:, None, :], dim=-1)) > thres)
            if pc.shape[-1] == 6:
                pc_normals = pc[:, 3:]
                normals_correlation = torch.sum(normals[:, None, :] * pc_normals, dim=-1)
                mask = mask * (normals_correlation > 0)
            torch.cuda.empty_cache()
            distance[~ mask] += float('inf')
            min, argmin = distance.min(dim=-1)

            pc_per_face_masked = pc[argmin, :].clone()
            pc_per_face_masked[min == float('inf'), :] = float('nan')
            pc_per_face = torch.zeros(mid_points.shape[0], 6).\
                type(pc_per_face_masked.dtype).to(pc_per_face_masked.device)
            pc_per_face[~ knn_mask, :pc.shape[-1]] = pc_per_face_masked
            pc_per_face[knn_mask, :] = float('nan')

            # clean up
            del knn_mask
        return pc_per_face.to(self.device), (pc_per_face[:, 0] == pc_per_face[:, 0]).to(device)

    @staticmethod
    def face_areas_normals(vs, faces):
        if type(vs) is not torch.Tensor:
            vs = torch.from_numpy(vs)
        if type(faces) is not torch.Tensor:
            faces = torch.from_numpy(faces)
        face_normals = torch.cross(vs[faces[:, 1]] - vs[faces[:, 0]],
                                   vs[faces[:, 2]] - vs[faces[:, 1]])

        face_areas = torch.norm(face_normals, dim=1)
        face_normals = face_normals / face_areas[:, None]
        face_areas = 0.5 * face_areas
        face_areas = 0.5 * face_areas
        return face_areas, face_normals

    def update_verts(self, verts):
        """
        update verts positions only, same connectivity
        :param verts: new verts
        """
        self.vs = verts

    def deep_copy(self): #TODO see if can do this better
        new_mesh = Mesh(file=None)
        types = [np.ndarray, torch.Tensor,  dict, list, str, int, bool, float]
        for attr in self.__dir__():
            if attr == '__dict__':
                continue

            val = getattr(self, attr)
            if type(val) == types[0]:
                new_mesh.__setattr__(attr, val.copy())
            elif type(val) == types[1]:
                new_mesh.__setattr__(attr, val.clone())
            elif type(val) in types[2:4]:
                new_mesh.__setattr__(attr, pickle.loads(pickle.dumps(val, -1)))
            elif type(val) in types[4:]:
                new_mesh.__setattr__(attr, val)

        return new_mesh

    def merge_vertices(self, edge_id):
        self.remove_edge(edge_id)
        edge = self.edges[edge_id]
        v_a = self.vs[edge[0]]
        v_b = self.vs[edge[1]]
        # update pA
        v_a.__iadd__(v_b)
        v_a.__itruediv__(2)
        self.v_mask[edge[1]] = False
        mask = self.edges == edge[1]
        self.ve[edge[0]].extend(self.ve[edge[1]])
        self.edges[mask] = edge[0]

    def remove_vertex(self, v):
        self.v_mask[v] = False

    def remove_edge(self, edge_id):
        vs = self.edges[edge_id]
        for v in vs:
            if edge_id not in self.ve[v]:
                print(self.ve[v])
                print(self.filename)
            self.ve[v].remove(edge_id)

    def clean(self, edges_mask, groups):
        edges_mask = edges_mask.astype(bool)
        torch_mask = torch.from_numpy(edges_mask.copy())
        self.gemm_edges = self.gemm_edges[edges_mask]
        self.edges = self.edges[edges_mask]
        self.sides = self.sides[edges_mask]
        new_ve = []
        edges_mask = np.concatenate([edges_mask, [False]])
        new_indices = np.zeros(edges_mask.shape[0], dtype=np.int32)
        new_indices[-1] = -1
        new_indices[edges_mask] = np.arange(0, np.ma.where(edges_mask)[0].shape[0])
        self.gemm_edges[:, :] = new_indices[self.gemm_edges[:, :]]
        for v_index, ve in enumerate(self.ve):
            update_ve = []
            # if self.v_mask[v_index]:
            for e in ve:
                update_ve.append(new_indices[e])
            new_ve.append(update_ve)
        self.ve = new_ve
        self.__clean_history(groups, torch_mask)

    def init_history(self):
        self.history_data = {
                               'groups': [],
                               'gemm_edges': [self.gemm_edges.copy()],
                               'occurrences': [],
                               'edges_count': [self.edges_count],
                              }

    def get_groups(self):
        return self.history_data['groups'].pop()

    def get_occurrences(self):
        return self.history_data['occurrences'].pop()
    
    def __clean_history(self, groups, pool_mask):
        if self.history_data is not None:
            self.history_data['occurrences'].append(groups.get_occurrences())
            self.history_data['groups'].append(groups.get_groups(pool_mask))
            self.history_data['gemm_edges'].append(self.gemm_edges.copy())
            self.history_data['edges_count'].append(self.edges_count)
    
    def unroll_gemm(self):
        self.history_data['gemm_edges'].pop()
        self.gemm_edges = self.history_data['gemm_edges'][-1]
        self.history_data['edges_count'].pop()
        self.edges_count = self.history_data['edges_count'][-1]

    @staticmethod
    def from_tensor(mesh, vs, faces, gfmm=True):
        mesh = Mesh(file=mesh.filename, vs=vs, faces=faces, device=mesh.device, hold_history=True, gfmm=gfmm)
        return mesh
