from functools import partial
import PIL.Image    
if not hasattr(PIL.Image, 'Transpose'):  
    PIL.Image.Transpose = PIL.Image   
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pyzjr.nn.models.bricks.drop import DropPath
from pyzjr.nn.models.bricks.initer import trunc_normal_
 
LayerNorm = partial(nn.LayerNorm, eps=1e-6)
 
class PatchPartition(nn.Module):
    def __init__(self, patch_size=4, in_channels=1, embed_dim=256, norm_layer=None):
        super().__init__()
        self.patch_size = (patch_size, patch_size, patch_size)
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_channels, self.embed_dim,
                              kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(self.embed_dim) if norm_layer else nn.Identity()
 
    def forward(self, x):
        _, _, H, W, D = x.shape
        pad_h = self.patch_size[0] - H % self.patch_size[0]
        pad_w = self.patch_size[1] - W % self.patch_size[1]
        pad_d = self.patch_size[2] - D % self.patch_size[2]
        x = F.pad(x, (0, pad_d, 0, pad_w, 0, pad_h))

        x = self.proj(x)     # [B, embed_dim, H/patch_size, W/patch_size]
        Wh, Ww, Wd = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)    # [B, num_patches, embed_dim]
        # Linear Embedding
        x = self.norm(x)
        return x, Wh, Ww, Wd
 
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop_ratio=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_ratio)
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)
 
    def forward(self, x, H, W, D):
        B, L, C = x.shape
        assert L == H * W * D, "input feature has wrong size"
 
        x = x.view(B, H, W, D, C)
        
        # Pad input if H/W/D are not even
        if H % 2 != 0 or W % 2 != 0 or D % 2 != 0:
            x = F.pad(x, (0, 0, 0, D % 2, 0, W % 2, 0, H % 2))
            H = H + (H % 2)
            W = W + (W % 2)
            D = D + (D % 2)
 
        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 D/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  
        x2 = x[:, 0::2, 1::2, 0::2, :]  
        x3 = x[:, 1::2, 1::2, 0::2, :]  
        x4 = x[:, 0::2, 0::2, 1::2, :]  
        x5 = x[:, 1::2, 0::2, 1::2, :]  
        x6 = x[:, 0::2, 1::2, 1::2, :]  
        x7 = x[:, 1::2, 1::2, 1::2, :]  
        
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 D/2 8*C
        x = x.view(B, -1, 8 * C)  # B (H/2*W/2*D/2) 8*C
 
        x = self.norm(x)
        x = self.reduction(x)
        return x
 
class WindowAttention(nn.Module):
#     """
#     Window based multi-head self attention (W-MSA) module with relative position bias.
#     It supports shifted and non-shifted windows.
#     """
    def __init__(
            self,
            dim,
            window_size,
            num_heads,
            qkv_bias=True,
            proj_bias=True,
            attention_dropout_ratio=0.,
            proj_drop=0.,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = (window_size, window_size, window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((3 * window_size - 1) ** 3, num_heads)
        )   # [2*Wh-1 * 2*Ww-1, nHeads]   Offset Range: -Wh+1, Wh-1
 
        self.register_buffer("relative_position_index",
                             self.get_relative_position_index(self.window_size[0], self.window_size[1], self.window_size[2]), persistent=False)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout_ratio)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
 
        self.softmax = nn.Softmax(dim=-1)
    
    def get_relative_position_index(self, win_d: int, win_h: int, win_w: int):
        # Create coordinate grid
        coords = torch.stack(torch.meshgrid(torch.arange(win_d), torch.arange(win_h), torch.arange(win_w),indexing='ij'))  # shape: [3, D, H, W]
        coords_flatten = torch.flatten(coords, 1)  # [3, D*H*W]
        # Compute relative coordinates
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [3, N, N]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [N, N, 3]

        # Shift to make all coordinates non-negative
        relative_coords[:, :, 0] += win_d - 1
        relative_coords[:, :, 1] += win_h - 1
        relative_coords[:, :, 2] += win_w - 1

        # Combine the 3D coordinates into a single unique index
        # Formula: z * (2H - 1)*(2W - 1) + y * (2W - 1) + x
        relative_coords[:, :, 0] *= (2 * win_h - 1) * (2 * win_w - 1)
        relative_coords[:, :, 1] *= (2 * win_w - 1)
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

 
    def forward(self, x, mask=None):
#         """
#         Args:
#             x: input features with shape of (num_windows*B, N, C)
#             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#         """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[:3]
 
        q = q * self.scale
        # print(q.shape)
        attn = (q @ k.transpose(-2, -1))
        # print(attn.shape)
 
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] ** 3, self.window_size[0] ** 3, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # print(attn.shape, relative_position_bias.shape)
        attn = attn + relative_position_bias.unsqueeze(0)
 
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
 
        attn = self.attn_drop(attn)
 
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
 
 
def window_partition(x, window_size: int):
#     """
#     Args:
#         x: (B, H, W, C)
#         window_size (int): window size(M)
#     Returns:
#         windows: (num_windows*B, window_size, window_size, C)
#     """
    B, H, W, D, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, D // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows
 
 
def window_reverse(windows, window_size: int, H: int, W: int, D: int):
#         """
#         Args:
#         windows: (num_windows*B, window_size, window_size, C)
#         window_size (int): Window size(M)
#         H (int): Height of image
#         W (int): Width of image
#         Returns:
#         x: (B, H, W, C)
#         """
    B = int(windows.shape[0] / (H * W * D / (window_size**3)))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, D // window_size, window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, D, -1)
    return x
 
class SwinTransformerBlock(nn.Module):
#     """ Swin Transformer Block."""
    mlp_ratio = 4
    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        qkv_bias=True,
        proj_bias=True,
        attention_dropout_ratio=0.,
        proj_drop=0.,
        drop_path_ratio=0.,
        norm_layer=LayerNorm,
        act_layer=nn.GELU,
    ):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < window_size, "shift_size must in 0-window_size"
 
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attention_dropout_ratio=attention_dropout_ratio,
            proj_drop=proj_drop,
        )
 
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * self.mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop_ratio=proj_bias)
 
        self.H = None
        self.W = None
        self.D = None
 
    def forward(self, x, mask_matrix):
#         """
#         Args:
#             x: Input feature, tensor size (B, H*W, C).
#             H, W: Spatial resolution of the input feature.
#             mask_matrix: Attention mask for cyclic shift.
#         """
        B, L, C = x.shape
        H, W, D = self.H, self.W, self.D
        assert L == H * W *D, "input feature has wrong size"
 
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, D, C)
 
        # pad feature maps to multiples of window size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_d = (self.window_size - D % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_d, 0, pad_w, 0, pad_h))
        _, Hp, Wp, Dp, _ = x.shape
 
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
 
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
 
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C
 
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp, Dp)  # B H' W' C
 
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
 
        if pad_w > 0 or pad_h > 0 or pad_d > 0:
            x = x[:, :H, :W, :D, :].contiguous()
 
        x = x.view(B, H * W * D, C)
 
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
 
        return x
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage."""
    def __init__(self,
                 dim,
                 num_layers,
                 num_heads,
                 drop_path,
                 window_size=7,
                 qkv_bias=True,
                 proj_bias=True,
                 attention_dropout_ratio=0.,
                 proj_drop=0.,
                 norm_layer=LayerNorm,
                 act_layer=nn.GELU,
                 downsample=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.num_layers = num_layers
 
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                attention_dropout_ratio=attention_dropout_ratio,
                proj_drop=proj_drop,
                drop_path_ratio=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                act_layer=act_layer)
            for i in range(num_layers)])
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
 
    def forward(self, x, H, W, D):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
 
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        Dp = int(np.ceil(D / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, Dp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        d_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                for d in d_slices:
                    img_mask[:, h, w, d, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
 
        for blk in self.blocks:
            blk.H, blk.W, blk.D = H, W, D
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x, H, W, D)
            H, W, D = (H + 1) // 2, (W + 1) // 2, (D+1) // 2
 
        return x, H, W, D
 
 
 
 
class SwinTransformer(nn.Module):
    """ Swin Transformer backbone."""
    def __init__(self,
                 patch_size=4,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 window_size=7,
                 qkv_bias=True,
                 proj_bias=True,
                 attention_dropout_ratio=0.,
                 proj_drop=0.,
                 drop_path_rate=0.2,
                 norm_layer=LayerNorm,
                 patch_norm=True,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
 
        # split image into non-overlapping patches
        self.patch_embed = PatchPartition(
            patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
 
        self.pos_drop = nn.Dropout(p=proj_drop)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        layers = []
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                num_layers=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                attention_dropout_ratio=attention_dropout_ratio,
                proj_drop=proj_drop,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                )
            layers.append(layer)
 
        self.layers = nn.Sequential(*layers)
 
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
 
        self._initialize_weights()
 
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
 
    def forward(self, x):
        # x: [B, L, C]
        x, H, W, D = self.patch_embed(x)
        x = self.pos_drop(x)
 
        for layer in self.layers:
            x, H, W, D = layer(x, H, W, D)
 
        x = self.norm(x)  # [B, L, C]
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x