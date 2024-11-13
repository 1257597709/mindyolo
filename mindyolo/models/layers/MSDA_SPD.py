import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from ..layers import ConvNormAct

# def autopad(k, p=None, d=1):  # kernel, padding, dilation
#     # Pad to 'same' shape outputs
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p


# class Conv(nn.Cell):
#     # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
#     default_act = nn.SiLU()  # default activation

#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         super(Conv, self).__init__()
#         padding = autopad(k, p, d)  # calculate padding using autopad function
#         self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=padding, group=g, dilation=d, has_bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Cell) else nn.Identity()

    # def construct(self, x):
    #     return self.act(self.bn(self.conv(x)))

    # def forward_fuse(self, x):
    #     # Fused forward method without BatchNorm (for inference)
    #     return self.act(self.conv(x))

class C2f_MSDA(nn.Cell):
    """CSP Bottleneck with 2 convolutions"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, momentum=0.97, eps=1e-3, sync_bn=False):
        super(C2f_MSDA, self).__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvNormAct(c1, 2 * self.c, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.cv2 = ConvNormAct((2 + n) * self.c, c2, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.m = nn.CellList([DilateBlock(self.c) for _ in range(n)])

        # self.concat = ops.Concat(axis=1)

    def construct(self, x):
        # y = ()
        # x = self.cv1(x)
        # _c = x.shape[1] // 2
        # x_tuple = ops.split(x, axis=1, split_size_or_sections=_c)
        # y += x_tuple
        # for i in range(len(self.m)):
        #     m = self.m[i]
        #     out = m(y[-1])
        #     y += (out,)

        # return self.cv2(ops.concat(y, axis=1))
        split_op = ops.Split(axis = 1, output_num = 2)
        y = list(split_op(self.cv1(x)))
        for m in self.m:
            y.append(m(y[-1]))
        concat_op = ops.Concat(axis = 1)
        
        return self.cv2(concat_op(y))

class DilateAttention(nn.Cell):
    """Implementation of Dilate-attention"""

    def __init__(self, head_dim, qk_scale=None, attn_drop=0., kernel_size=3, dilation=1):
        super(DilateAttention, self).__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        # maybe different
        self.unfold = nn.Unfold(ksizes=[1,kernel_size, kernel_size,1], 
                                strides=[1, 1, 1, 1], 
                                rates=[1, 1, 1, 1], padding="same")

        self.attn_drop = nn.Dropout(p=attn_drop)

    def construct(self, q, k, v):
        B, d, H, W = q.shape
        q = q.view(B, d // self.head_dim, self.head_dim, 1, H * W).transpose(0, 1, 4, 3, 2)
        k = self.unfold(k).view(B, d // self.head_dim, 
                                self.head_dim, 
                                self.kernel_size * self.kernel_size, 
                                H * W).transpose(0, 1, 4, 2, 3)
        attn = ops.matmul(q, k) * self.scale
        attn = ops.Softmax(axis=-1)(attn)
        attn = self.attn_drop(attn)
        v = self.unfold(v).view(B, d // self.head_dim,
                                self.head_dim, 
                                self.kernel_size * self.kernel_size, 
                                H * W).transpose(0, 1, 4, 3, 2)
        x = ops.matmul(attn, v).transpose(0, 2, 1, 3, 4).view(B, H, W, d)
        return x

class MultiDilatelocalAttention(nn.Cell):
    """Implementation of Dilate-attention"""

    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2]):
        super(MultiDilatelocalAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)

        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be divisible by num_dilation{self.num_dilation}!!"
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, has_bias=qkv_bias)
        self.dilate_attention = nn.CellList([
            DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i]) 
            for i in range(self.num_dilation)])
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def construct(self, x):
        x = ops.Transpose()(x, (0, 3, 1, 2))  # B, C, H, W
        B, C, H, W = x.shape

        qkv = self.qkv(x).view(B, 3, self.num_dilation, C // self.num_dilation, H, W).transpose(2, 1, 0, 3, 4, 5)
        x = x.view(B, self.num_dilation, C // self.num_dilation, H, W).transpose(1, 0, 3, 4, 2)

        for i in range(self.num_dilation):
            x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])
        
        x = ops.Transpose()(x, (1, 2, 3, 0, 4)).view(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DilateBlock(nn.Cell):
    """Implementation of Dilate-attention block"""

    def __init__(self, dim, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0, 
    drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=3, dilation=[1, 2], cpe_per_block=False):
        super(DilateBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.cpe_per_block = cpe_per_block

        if self.cpe_per_block:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=3, padding=1, group=dim)
        
        self.norm1 = norm_layer([dim])
        self.attn = MultiDilatelocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
        qk_scale=qk_scale, attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation)
        self.drop_path = nn.Identity()

    def construct(self, x):
        x = ops.Transpose()(x, (0, 3, 2, 1))
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = ops.Transpose()(x, (0, 3, 2, 1))
        return x


class space_to_depth(nn.Cell):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        concat = ops.Concat(axis=1)
    # 使用 MindSpore ops 实现类似的操作
        output = concat([x[..., ::2, ::2], x[..., 1::2, ::2], 
                        x[..., ::2, 1::2], x[..., 1::2, 1::2]])
        return output
