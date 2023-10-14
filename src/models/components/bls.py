from typing import List, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from torchvision.utils import _log_api_usage_once


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


class SwiftFormerEncoder(nn.Module):
    """SwiftFormer Encoder Block for SwiftFormer.

    It consists of (1) Local representation module, (2) EfficientAdditiveAttention, and (3) MLP block.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]

    Args:
        dim (int): The number of input channels.
        mlp_ratio (float): The ratio of the hidden size of the MLP layer to the input size.
        act_layer (nn.Module): The activation function to use in the MLP layer.
        drop (float): The dropout probability to use in the MLP layer.
        drop_path (float): The dropout probability to use in the residual connection.
        use_layer_scale (bool): Whether to use layer scale in the residual connection.
        layer_scale_init_value (float): The initial value of the layer scale parameter.
    """

    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        drop=0.0,
        drop_path=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
    ):
        super().__init__()

        self.local_representation = SwiftFormerLocalRepresentation(
            dim=dim, kernel_size=3, drop_path=0.0, use_layer_scale=True
        )
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
        """Additive encoder block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
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


class h_sigmoid(nn.Module):
    """Applies the hard sigmoid function element-wise.

    The hard sigmoid function is defined as:
    relu(x + 3) / 6

    Args:
        inplace (bool, optional): If set to True, will modify the input tensor in-place. Default: True.
    """

    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        """Applies the hard sigmoid function element-wise."""
        return self.relu(x + 3) / 6


class ECALayer(nn.Module):
    """Effective Channel Attention."""

    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = h_sigmoid()

    def forward(self, x):
        """ECA layer is very close to MLP but with a representation to 1d form."""
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


def conv1x1(in_planes: int, out_planes: int, **kwargs) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, **kwargs)


def conv3x3(in_planes: int, out_planes: int, **kwargs) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, **kwargs)


def conv5x5(in_planes: int, out_planes: int, **kwargs) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, padding=2, **kwargs)


class FeatureBlock(nn.Module):
    """BLS feature block.

    This class defines a feature block for the BLS model. It consists of a series of convolutional layers
    followed by a downsample operation and a dropout layer. The block can be used multiple times in the
    model architecture.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        dropout_rate (float): Dropout rate to use.
        conv_cnt (int): Number of convolutional layers to use.
        down_gap (int): Stride to use for the downsample operation.
        islast (bool): Whether this is the last feature block in the model.

    Returns:
        Output tensor after passing through the feature block.
    """

    def __init__(
        self,
        inplanes: int,
        planes: int,
        dropout_rate: float = 0.1,
        conv_cnt: int = 3,
        down_gap: int = 2,
        islast: bool = False,
    ) -> None:
        super().__init__()

        # print(f'FB parameters: {inplanes} -> {planes}, with {conv_cnt} convs')

        self.extractor = nn.ModuleList(
            [
                nn.Sequential(
                    conv3x3(inplanes if _ == 0 else planes, planes, bias=False),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate),
                )
                for _ in range(conv_cnt - 1)
            ]
        )

        # if this is the last block, then use 5x5 conv with stride 2
        head_conv = conv3x3 if not islast else conv5x5

        # Add downsample operation to match input and output dimensions
        downsample = (
            nn.Conv2d(planes, planes, kernel_size=down_gap, stride=down_gap)
            if not islast
            else nn.Identity()
        )
        # downsample = nn.AvgPool2d(kernel_size=down_gap) if not islast else nn.Identity()

        self.head = nn.Sequential(
            head_conv(inplanes if conv_cnt == 1 else planes, planes, bias=False),
            nn.ReLU(inplace=True),
            ECALayer(k_size=3),
            downsample,
            nn.Dropout(dropout_rate),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass of the BLS model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        for _, layer in enumerate(self.extractor):
            x = layer(x)
        x = self.head(x)
        return x


class GDBLS(nn.Module):
    """GDBLS is a neural network model for image classification tasks. It consists of a series of
    feature extraction blocks, followed by two layers of SwiftFormer encoders, and a final
    classification head. The number of feature extraction blocks, their depth, and the number of
    filters in each block can be customized. The input images are expected to have shape
    [batch_size, 3, height, width], and the output is a tensor of shape [batch_size, num_classes],
    where num_classes is the number of classes to predict.

    Args:
        num_classes (int): number of classes to predict.
        input_shape (List[int]): shape of the input images, as a list of integers [channels, height, width].
        fb_cnt (int): number of feature extraction blocks.
        fb_depth (int): depth of each feature extraction block.
        filters (List[int]): number of filters in each feature extraction block.
        block_dropout (List[float]): dropout rate for each feature extraction block.
        overall_dropout (float): dropout rate for the final classification head.
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_shape: List[int] = [3, 32, 32],
        fb_cnt=3,
        fb_depth=3,
        filters: List[int] = [64, 128, 256],
        block_dropout: List[float] = [0.5, 0.5, 0.5],
        overall_dropout: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)

        assert len(filters) == fb_cnt
        assert len(block_dropout) == fb_cnt

        self.fb_cnt = fb_cnt
        self.num_classes = num_classes
        self.C, self.H, self.W = input_shape

        # init downsample gap, this is used in Pooling kernel
        if self.H <= 128:
            self.gap = 2
        else:
            self.gap = 4

        self.input_layer = nn.Sequential(
            conv3x3(self.C, 64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.extractors = nn.ModuleList(
            FeatureBlock(
                inplanes=64 if i == 0 else filters[i - 1],
                planes=filters[i],
                dropout_rate=block_dropout[i],
                conv_cnt=fb_depth,
                islast=False if i != fb_cnt - 1 else True,
                down_gap=self.gap,
            )
            for i in range(fb_cnt)
        )

        self.encoders_l1 = nn.ModuleList(
            SwiftFormerEncoder(
                dim=filters[i],
                mlp_ratio=4.0,
                drop=block_dropout[i],
            )
            for i in range(fb_cnt)
        )

        # align map size and channels to the last one
        self.aligners = nn.ModuleList(
            nn.Sequential(
                nn.AvgPool2d(kernel_size=self.gap ** (fb_cnt - i - 2)),
                conv1x1(
                    in_planes=filters[i],
                    out_planes=filters[fb_cnt - 1],
                    bias=False,
                ),
            )
            for i in range(fb_cnt - 1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(inplace=True),
            nn.Dropout(overall_dropout),
            nn.Linear(filters[-1], self.num_classes),
        )

        self.init_weights()

    def init_weights(self):
        """Initializes the weights of the convolutional and batch normalization layers using the
        Kaiming normal initialization for the convolutional layers and setting the weights of the
        batch normalization layers to 1 and the biases to 0."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward_features(self, x):
        """Forward pass through the BLS model's feature extraction and encoding layers.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            List[torch.Tensor]: List of feature maps extracted at each scale of the BLS model.
        """
        ps = []
        for i in range(self.fb_cnt):
            x = self.extractors[i](x if i == 0 else ps[i - 1])
            x = self.encoders_l1[i](x)
            # x = self.encoders_l2[i](x)
            ps.append(x)
        return ps

    def forward(self, x) -> torch.Tensor:
        """Forward pass of the BLS model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.input_layer(x)
        ps = self.forward_features(x)
        ps = [self.aligners[i](x) for i, x in enumerate(ps[:-1])]
        out = self.avgpool(sum(ps))
        out = self.head(out)
        return out
