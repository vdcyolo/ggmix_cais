def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='border',
              align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    device = flow.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=device, dtype=x.dtype),
        torch.arange(0, w, device=device, dtype=x.dtype))
    grid = torch.stack((grid_x, grid_y), 2)  # h, w, 2
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
        self.eps = eps
        self.data_format = data_format
        
        normalized_dim = len(self.normalized_shape)
        param_shape = self.normalized_shape
        
        self.gamma = nn.Parameter(torch.ones(*param_shape) * 1.0)
        self.beta = nn.Parameter(torch.zeros(*param_shape))
        
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"Unsupported data format: {self.data_format}")
    
    def forward(self, x):
        if self.data_format == "channels_last":
            # 使用F.layer_norm进行channels_last层归一化
            return F.layer_norm(x, self.normalized_shape, self.gamma, self.beta, self.eps)

        channel_dim = 1

        mean = x.mean(dim=channel_dim, keepdim=True)

        var = x.var(dim=channel_dim, keepdim=True, unbiased=False)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        reshaped_gamma = self.gamma.view((-1,) + (1,) * (x.ndim - 2))
        reshaped_beta = self.beta.view((-1,) + (1,) * (x.ndim - 2))
        
        return x_normalized * reshaped_gamma + reshaped_beta



class Global_Guidance(nn.Module):
    def __init__(self, dim, window_size=4, k=4,ratio=0.5):
        super().__init__()

        self.ratio = ratio
        self.window_size = window_size
        cdim = dim + k
        embed_dim = window_size**2

        self.in_conv = nn.Sequential(
            nn.Conv2d(cdim, cdim//4, 1),
            LayerNorm(cdim//4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.out_offsets = nn.Sequential(
            nn.Conv2d(cdim//4, cdim//8, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(cdim//8, 2, 1),
        )

        self.out_mask = nn.Sequential(
            nn.Linear(embed_dim, window_size),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(window_size, 2),
            nn.Softmax(dim=-1)
        )

        self.out_CA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cdim//4, dim, 1),
            nn.Sigmoid(),
        )


        self.out_SA = nn.Sequential(
            nn.Conv2d(cdim//4, 1, 3, 1, 1),
            nn.Sigmoid(),
        )        


    def forward(self, input_x, mask=None, ratio=0.5, train_mode=False):
        x = self.in_conv(input_x)

        offsets = self.out_offsets(x)
        offsets = offsets.tanh().mul(8.0)

        ca = self.out_CA(x)
        sa = self.out_SA(x)
        
        x = torch.mean(x, keepdim=True, dim=1) 

        x = rearrange(x,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        B, N, C = x.size()

        pred_score = self.out_mask(x)
        mask = F.gumbel_softmax(pred_score, hard=True, dim=2)[:, :, 0:1]

        return mask, offsets, ca, sa


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q, k, v):
        B, N, C = q.shape

        q = self.q_proj(q).reshape(B, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        k = self.k_proj(k).reshape(B, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        v = self.v_proj(v).reshape(B, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bottleneck_ratio=0.5, groups=4):

        super().__init__()
        bottleneck_channels = int(in_channels * bottleneck_ratio)

        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels
        )
        
        self.bottleneck = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0)
        
        self.pointwise = nn.Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bottleneck(x)
        x = self.pointwise(x)
        return x
    

class GGmix(nn.Module):
    def __init__(self, dim, window_size=4, bias=True, is_deformable=True, ratio=0.5):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.is_deformable = is_deformable
        self.ratio = ratio

        k = 3
        d = 2

        self.project_v = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)
        self.project_q = nn.Linear(dim, dim, bias=bias)
        self.project_k = nn.Linear(dim, dim, bias=bias)

        self.multihead_attn = MultiHeadAttention(dim)
        self.conv_sptial = DepthwiseSeparableConv(dim, dim)   
        self.project_out = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)

        self.act = nn.GELU()
        self.route = Global_Guidance(dim, window_size, ratio=ratio)

        self.global_predictor = nn.Sequential(
            nn.Conv2d(3, 8, 1, 1, 0, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(8, dim+2, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x, condition_global=None, mask=None, train_mode=True, img_ori=None):
        N, C, H, W = x.shape

        global_status = self.global_predictor(img_ori)
        global_status = F.interpolate(global_status, size=(H, W), mode='bilinear', align_corners=False)

        if self.is_deformable:
            patch_status = torch.stack(torch.meshgrid(torch.linspace(-1, 1, self.window_size), torch.linspace(-1, 1, self.window_size)))\
                .type_as(x).unsqueeze(0).repeat(N, 1, H // self.window_size, W // self.window_size)
            
            global_feature = torch.cat([global_status, patch_status], dim=1)
            
        mask, offsets, ca, sa = self.route(global_feature, ratio=self.ratio, train_mode=train_mode)

        q = x
        k = x + flow_warp(x, offsets.permute(0, 2, 3, 1))
        qk = torch.cat([q, k], dim=1)
        v = self.project_v(x)

        vs = v * sa

        v = rearrange(v, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        vs = rearrange(vs, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        qk = rearrange(qk, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)

        N_ = v.shape[1]
        v1, v2 = v * mask, vs * (1 - mask)
        qk1 = qk * mask
    

        v1 = rearrange(v1, 'b n (dh dw c) -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        qk1 = rearrange(qk1, 'b n (dh dw c) -> b (n dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)

        q1, k1 = torch.chunk(qk1, 2, dim=2)
        q1 = self.project_q(q1)
        k1 = self.project_k(k1)
        q1 = rearrange(q1, 'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        k1 = rearrange(k1, 'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)

        f_attn = self.multihead_attn(q1, k1, v1)

        f_attn = rearrange(f_attn, '(b n) (dh dw) c -> b n (dh dw c)',
            b=N, n=N_, dh=self.window_size, dw=self.window_size)

        
        attn_out = f_attn + v2

        attn_out = rearrange(
            attn_out, 'b (h w) (dh dw c) -> b (c) (h dh) (w dw)',
            h=H // self.window_size, w=W // self.window_size, dh=self.window_size, dw=self.window_size
        )

        out = attn_out
        out = self.act(self.conv_sptial(out)) * ca + out
        out = self.project_out(out)

        return out
