import math

import einops
import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn


def conv1x1(in_planes, out_planes):
    "custom 1x1 conv layer"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)


def conv3x3(in_planes, out_planes):
    "custom 3x3 conv layer"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, bias=False)


class PLVPooling(nn.Module):
    """PLVPooling Layer."""

    def __init__(self, channels_first=True, **kwargs):
        super().__init__()
        self.dims = [2, 3] if channels_first else [1, 2]

    def forward(self, x, bias):
        _, C, H, W = x.shape
        ppvoutput = torch.mean(
            torch.greater(x, torch.reshape(bias, (C, 1, 1)).expand((C, H, W))).float(),
            dim=self.dims,
        )
        return ppvoutput


class EfficientAdditiveAttnetion(nn.Module):
    """Efficient Additive Attention module for SwiftFormer.

    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """

    def __init__(self, in_dims=512, token_dim=256, num_heads=2):
        super().__init__()

        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim**-0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        """Additive attention implation."""
        query = self.to_query(x)
        key = self.to_key(x)

        query = torch.nn.functional.normalize(query, dim=-1)  # BxNxD
        key = torch.nn.functional.normalize(key, dim=-1)  # BxNxD

        query_weight = query @ self.w_g  # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor  # BxNx1

        A = torch.nn.functional.normalize(A, dim=1)  # BxNx1

        G = torch.sum(A * query, dim=1)  # BxD

        G = einops.repeat(G, "b d -> b repeat d", repeat=key.shape[1])  # BxNxD

        out = self.Proj(G * key) + query  # BxNxD

        out = self.final(out)  # BxNxD

        return out


class SwiftFormerLocalRepresentation(nn.Module):
    """Local Representation module for SwiftFormer that is implemented by 3*3 depth-wise and point-
    wise convolutions.

    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """

    def __init__(self, dim, kernel_size=3, drop_path=0.0, use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True
            )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Init weights for conv2d layer.

        bias for conv2d layer is set to zero.
        """
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Local representation."""
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x


class Mlp(nn.Module):
    """Implementation of MLP layer with 1*1 convolutions.

    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = nn.BatchNorm2d(in_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Init weights for conv2d layer.

        bias for conv2d layer is set to zero.
        """
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Mlp forward function.

        classicial two layer structure, normalize input first and fc-act-drop-fc-drop.
        """
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        drop=0.0,
        drop_path=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
    ) -> None:
        """Initializes a BLSv1 TransformerEncoder block.

        Args:
        - dim (int): the number of input channels
        - mlp_ratio (float): the ratio of the hidden MLP dimension to the input dimension (default: 4.0)
        - act_layer (nn.Module): the activation function to be used (default: nn.GELU)
        - drop (float): the dropout probability (default: 0.0)
        - drop_path (float): the drop path probability (default: 0.0)
        - use_layer_scale (bool): whether to use layer scale (default: True)
        - layer_scale_init_value (float): the initial value of the layer scale (default: 1e-5)
        """
        super().__init__()
        self.local_representation = SwiftFormerLocalRepresentation(
            dim=dim, kernel_size=3, drop_path=0.0, use_layer_scale=True
        )
        # self.attn = FlashAttention(dim=dim, heads=1, dim_head=dim)
        self.attn = EfficientAdditiveAttnetion(in_dims=dim, token_dim=dim, num_heads=1)
        self.linear = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1),
                requires_grad=True,
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1),
                requires_grad=True,
            )

    def forward(self, x):
        x = self.local_representation(x)
        B, C, H, W = x.shape
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1
                * self.attn(x.permute(0, 2, 3, 1).reshape(B, H * W, C))
                .reshape(B, H, W, C)
                .permute(0, 3, 1, 2)
            )
            x = x + self.drop_path(self.layer_scale_2 * self.linear(x))

        else:
            x = x + self.drop_path(
                self.attn(x.permute(0, 2, 3, 1).reshape(B, H * W, C))
                .reshape(B, H, W, C)
                .permute(0, 3, 1, 2)
            )
            x = x + self.drop_path(self.linear(x))
        return x


class FeatureBlock(nn.Module):
    """"""

    def __init__(self, inplanes, planes, num_layers=4, dropout_rate=0.1):
        super().__init__()

        assert num_layers > 0

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    conv3x3(inplanes if i == 0 else planes, planes),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate),
                )
                for i in range(num_layers)
            ]
        )

        self.convlast = conv3x3(planes, planes, bias=True)
        self.relu5 = nn.ReLU(inplace=True)

        self.pool = PLVPooling()
        self.downsample = nn.AvgPool2d(kernel_size=2)

        # Add transformer block
        self.transformer = TransformerEncoder(dim=planes, mlp_ratio=2, use_layer_scale=True)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        # channel attention
        identity = x
        B, C, H, W = x.shape
        seout = self.pool(x, self.convlast.bias).reshape(B, C, 1, 1)
        x = seout * identity

        # self attention
        x = self.transformer(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class GDBLS(nn.Module):
    def __init__(
        self,
        num_classes=10,
        input_channels=3,
        filters=[64, 128, 256],
        num_layers=4,
        dropout_rate=0.5,
        block_drop_rate=0.1,
    ):
        super().__init__()

        self.num_classes = num_classes

        fb_cnt = len(filters)
        self.feature_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    FeatureBlock(
                        inplanes=input_channels if i == 0 else filters[i - 1],
                        planes=filters[i],
                        num_layers=num_layers,
                        dropout_rate=block_drop_rate,
                    )
                )
                for i in range(fb_cnt)
            ]
        )

        self.align = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AvgPool2d(kernel_size=2 ** (fb_cnt - i - 1)),
                    conv1x1(filters[i], filters[-1]),
                )
                if i != fb_cnt - 1
                else nn.Identity()
                for i in range(fb_cnt)
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(filters[-1], num_classes))

    def forward(self, x):
        features = [x]

        # forward & align features
        for ind, fb in enumerate(self.feature_blocks):
            features.append(fb(features[ind]))
        features = features[1:]
        features = [aligner(feat) for aligner, feat in zip(self.align, features)]

        # run classification
        feature = self.pool(sum(features)).flatten(start_dim=1)
        out = self.head(feature)
        return out
