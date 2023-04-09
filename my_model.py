import torch
import torch.nn as nn


class PatchPartition(nn.Module):
    def __init__(self,
                 patch_size: int = 4,
                 ):
        """
        this patch partition + Linear Embedding
        :param patch_size:
        """
        super().__init__()
        self.proj = nn.Conv2d(3, 96, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(96)

    def forward(self, x):
        x = self.proj(x)                  # [B, 96, 56, 56]
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self,
                 patch_size: int = 4,
                 ):
        """
        this patch partition + Linear Embedding
        :param patch_size:
        """
        super().__init__()
        self.proj = nn.Conv2d(3, 96, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(96)

    def forward(self, x):
        x = self.proj(x)                  # [B, 96, 56, 56]
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# class W_MSA(nn.Module):
#     def __init__(self, dim, heads, head_dim, window_size):
#         super().__init__()
#         inner_dim = head_dim * heads
#
#         self.heads = heads
#         self.scale = head_dim ** -0.5
#         self.window_size = window_size
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
#         # self.relative_indices = get_relative_distances(window_size) + window_size - 1
#         self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
#         self.to_out = nn.Linear(inner_dim, dim)
#
#     def forward(self, x):
#         b, n_h, n_w, _, h = *x.shape, self.heads
#
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         nw_h = n_h // self.window_size
#         nw_w = n_w // self.window_size
#
#         q, k, v = map(
#             lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
#                                 h=h, w_h=self.window_size, w_w=self.window_size), qkv)
#
#         dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale
#         dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
#         attn = dots.softmax(dim=-1)
#
#         out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
#         out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
#                         h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
#         out = self.to_out(out)
#         return out


class W_MSA(nn.Module):
    def __init__(self,
                 dim, num_heads, head_dim=None, window_size=7,
                 qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        win_h, win_w = (window_size, window_size)
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinBlock(nn.Module):

    def __init__(self,
                 dim: int = 96,
                 num_heads: int = 3,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 window_size: int = 7,
                 height_width: tuple = (56, 56)):
        super().__init__()
        #
        self.ws = window_size
        self.hw = height_width

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.w_msa = W_MSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.sw_msa = W_MSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.mlp1 = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.mlp2 = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        """
        window size is 7
        :param x: tensor - [B, L, C]
        :return:
        """
        # set window size is 7 and h, w
        ws = self.ws
        h, w = self.hw
        h_ = h // ws
        w_ = w // ws
        B, L, C = x.shape

        # efficient batch computation for shifted configuration
        res = x = self.norm1(x)                       # [B, 3136, C]

        # ------------------------------------------------------------------------------------------
        x = x.view(B, h, w, C)                        # [B, H, W, C]
        x = x.view(B, h_, ws, w_, ws, C)              # [0, 1, 2, 3, 4, 5 ] -> [0, 1, 3, 2, 4, 5 ] - idx
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B, 8, 7, 8, 7, 96] -> [B, 8, 8, 7, 7, 96]
        x = x.view(B * h_ * w_, ws * ws, C)           # [B' = B x 8 x 8],   -> [B'         49, 96]
        # following attention operation is computed in parallel through batches
        x = self.w_msa(x)                             # [B'         49, 96]
        x = x.view(B, h_, w_, ws, ws, C)              # [B, 8, 8, 7, 7, 96]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B, 8, 7, 8, 7, 96]
        x = x.view(B, h, w, -1)               # (roll)  [B, 56, 56, 96]
        x = x.view(B, h * w, C)                       # [B, 56, 56, 96]
        # ------------------------------------------------------------------------------------------

        x = res + self.w_msa(x)                       # [B, 3136, 96]
        x = x + self.mlp1(self.norm2(x))               # [B, 3136, 96]
        return x


class SwinTransformer(nn.Module):
    def __init__(self, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24)):
        super().__init__()
        self.patch_partition = PatchPartition()
        self.swin_block = SwinBlock()

    def forward(self, x):
        x = self.patch_partition(x)       # [2, 3136, 96]
        x = self.swin_block(x)
        return x


if __name__ == '__main__':
    img = torch.randn([2, 3, 224, 224])
    model = SwinTransformer()
    print(model)
    print(model(img).size())
