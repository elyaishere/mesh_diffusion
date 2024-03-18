import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from models.layers.mesh_conv import MeshConv
from models.layers.mesh_pool import MeshPool
from models.layers.mesh_unpool import MeshUnpool


class ConvBlock(nn.Module):
    def __init__(self, in_feat, out_feat, k=1):
        super(ConvBlock, self).__init__()
        self.lst = [MeshConv(in_feat, out_feat)] + [MeshConv(out_feat, out_feat) for _ in range(k - 1)]
        self.lst = nn.ModuleList(self.lst)

    def forward(self, input, meshes):
        for c in self.lst:
            input = c(input, meshes)
        return input


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, te_dim=None, blocks=0, pool=0):
        super(DownConv, self).__init__()
        self.bn = []
        self.pool = None
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(ConvBlock(out_channels, out_channels))
        self.conv2 = nn.ModuleList(self.conv2)
        for _ in range(blocks + 1):
            self.bn.append(nn.InstanceNorm2d(out_channels))
        self.bn = nn.ModuleList(self.bn)
        if pool:
            self.pool = MeshPool(pool)

        if te_dim:
            self.te = nn.Sequential(
                nn.SiLU(inplace=True),
                nn.Linear(te_dim, out_channels)
            )

    def forward(self, x, t=None):
        fe, meshes = x
        x1 = self.conv1(fe, meshes)
        if t is not None:
            t = self.te(t)
            t = einops.rearrange(t, "b c -> b c 1 1")
            x1 += t
        x1 = F.silu(x1, inplace=True)
        if self.bn:
            x1 = self.bn[0](x1)
        x2 = x1
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1, meshes)
            x2 = F.silu(x2, inplace=True)
            if self.bn:
                x2 = self.bn[idx + 1](x2)
            x2 = x2 + x1
            x1 = x2
        x2 = x2.squeeze(3)
        before_pool = None
        if self.pool:
            before_pool = x2
            x2 = self.pool(x2, meshes)
        return x2, before_pool


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, te_dim=None, blocks=0, unroll=0, residual=True,
                 batch_norm=True, transfer_data=True):
        super(UpConv, self).__init__()
        self.residual = residual
        self.bn = []
        self.unroll = None
        self.transfer_data = transfer_data
        self.up_conv = ConvBlock(in_channels, out_channels)
        if transfer_data:
            self.conv1 = ConvBlock(2 * out_channels, out_channels)
        else:
            self.conv1 = ConvBlock(out_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(ConvBlock(out_channels, out_channels))
        self.conv2 = nn.ModuleList(self.conv2)
        if batch_norm:
            for _ in range(blocks + 1):
                self.bn.append(nn.InstanceNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if unroll:
            self.unroll = MeshUnpool(unroll)
        
        if te_dim:
            self.te = nn.Sequential(
                nn.SiLU(inplace=True),
                nn.Linear(te_dim, out_channels)
            )

    def forward(self, x, from_down=None, t=None):
        from_up, meshes = x
        x1 = self.up_conv(from_up, meshes).squeeze(3)
        if self.unroll:
            x1 = self.unroll(x1, meshes)
        if self.transfer_data:
            x1 = torch.cat((x1, from_down), 1)
        
        x1 = self.conv1(x1, meshes)
        if t is not None:
            t = self.te(t)
            t = einops.rearrange(t, "b c -> b c 1 1")
            x1 += t
        x1 = F.silu(x1, inplace=True)
        if self.bn:
            x1 = self.bn[0](x1)

        x2 = x1
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1, meshes)
            x2 = F.silu(x2, inplace=True)
            if self.bn:
                x2 = self.bn[idx + 1](x2)
            if self.residual:
                x2 = x2 + x1
            x1 = x2
        x2 = x2.squeeze(3)
        return x2
