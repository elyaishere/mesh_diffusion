import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import init

from models.layers.blocks import UpConv, DownConv
from models.layers.mesh_conv import MeshConv


"""
https://github.com/ranahanocka/point2mesh/blob/333dba0b2ced97adfbdb62a5383d04bb5628680b/models/networks.py
"""


def build_v(x, meshes):
    # mesh.edges[mesh.ve[2], mesh.vei[2]]
    mesh = meshes[0]  # b/c all meshes in batch are same
    x = x.reshape(len(meshes), 2, 3, -1)
    vs_to_sum = torch.zeros([len(meshes), len(mesh.vs_in), mesh.max_nvs, 3], dtype=x.dtype, device=x.device)
    x = x[:, mesh.vei, :, mesh.ve_in].transpose(0, 1)
    vs_to_sum[:, mesh.nvsi, mesh.nvsin, :] = x
    vs_sum = torch.sum(vs_to_sum, dim=2)
    nvs = mesh.nvs
    vs = vs_sum / nvs[None, :, None]
    return vs


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # Assume time is a tensor with shape (batch_size, 1)
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class UNet(nn.Module):
    """
    network for
    """
    def __init__(self, n_edges, in_ch=6, convs=[16, 32, 64, 64, 128], pool=[0.0, 0.0, 0.0, 0.0], res_blocks=3,
                 transfer_data=False, init_weights_size=0.002, n_steps=1000):
        super(UNet, self).__init__()
        # check that the number of pools and convs match such that there is a pool between each conv
        down_convs = [in_ch] + convs
        up_convs = convs[::-1] + [in_ch]
        pool = [int(n_edges - i) for i in [i * n_edges for i in pool]]
        if (np.array(pool) == n_edges).all():
            pool = []
        pool_res = [n_edges] + pool
        time_dim = convs[0] * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(convs[0]),
            nn.Linear(convs[0], time_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_dim, time_dim),
        )
        self.encoder_decoder = MeshEncoderDecoder(pools=pool_res, down_convs=down_convs,
                                                  up_convs=up_convs, blocks=res_blocks,
                                                  transfer_data=transfer_data, te_dim=time_dim)
        self.last_conv = MeshConv(6, 6)
        init_weights(self, init_weights_size)
        eps = 1e-8
        self.last_conv.conv.weight.data.uniform_(-1*eps, eps)
        self.last_conv.conv.bias.data.uniform_(-1*eps, eps)

    def forward(self, x, meshes, t):
        meshes_new = [i.deep_copy() for i in meshes]
        t = self.time_mlp(t)
        x, _ = self.encoder_decoder(x, meshes_new, t)
        x = x.squeeze(-1)
        x = self.last_conv(x, meshes_new).squeeze(-1)
        est_verts = build_v(x.unsqueeze(0), meshes)
        assert not torch.isnan(est_verts).any()
        return est_verts.float()  # displacements


class MeshDecoder(nn.Module):
    def __init__(self, unrolls, convs, te_dim=None, blocks=0, batch_norm=True, transfer_data=True):
        super(MeshDecoder, self).__init__()
        self.up_convs = []
        for i in range(len(convs) - 2):
            if i < len(unrolls):
                unroll = unrolls[i]
            else:
                unroll = 0
            self.up_convs.append(UpConv(convs[i], convs[i + 1], te_dim=te_dim, blocks=blocks, unroll=unroll,
                                        batch_norm=batch_norm, transfer_data=transfer_data))
        self.final_conv = UpConv(convs[-2], convs[-1], blocks=blocks, unroll=False,
                                 batch_norm=batch_norm, transfer_data=False)
        self.up_convs = nn.ModuleList(self.up_convs)
        reset_params(self)

    def forward(self, x, encoder_outs=None, t=None):
        fe, meshes = x
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i+2)]
            fe = up_conv((fe, meshes), before_pool, t=t)
        fe = self.final_conv((fe, meshes))
        return fe

    def __call__(self, x, encoder_outs=None, t=None):
        return self.forward(x, encoder_outs, t=None)



class MeshEncoder(nn.Module):
    def __init__(self, pools, convs, te_dim=None, blocks=0):
        super(MeshEncoder, self).__init__()
        self.convs = []
        for i in range(len(convs) - 1):
            if i + 1 < len(pools):
                pool = pools[i + 1]
            else:
                pool = 0
            self.convs.append(DownConv(convs[i], convs[i + 1], te_dim=te_dim, blocks=blocks, pool=pool))
        self.convs = nn.ModuleList(self.convs)
        reset_params(self)

    def forward(self, x, t=None):
        fe, meshes = x
        encoder_outs = []
        for conv in self.convs:
            fe, before_pool = conv((fe, meshes), t=t)
            encoder_outs.append(before_pool)
        return fe, encoder_outs


class MeshEncoderDecoder(nn.Module):
    """Network for fully-convolutional tasks (segmentation)
    """
    def __init__(self, pools, down_convs, up_convs, te_dim, blocks=0, transfer_data=True):
        super(MeshEncoderDecoder, self).__init__()
        self.transfer_data = transfer_data
        self.encoder = MeshEncoder(pools, down_convs, te_dim=te_dim, blocks=blocks)
        unrolls = pools[:-1].copy()
        unrolls.reverse()
        self.decoder = MeshDecoder(unrolls, up_convs, te_dim=te_dim, blocks=blocks, transfer_data=transfer_data)
        self.bn = nn.InstanceNorm2d(up_convs[-1])

    def forward(self, x, meshes, t):
        fe, before_pool = self.encoder((x, meshes), t=t)
        fe = self.decoder((fe, meshes), before_pool, t=t)
        fe = self.bn(fe.unsqueeze(-1))
        return fe, None
    

def reset_params(model):
    for m in model.modules():
        weight_init(m)


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def init_weights(net, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, init_gain)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)
