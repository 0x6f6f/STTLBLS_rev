# basic dependicies
import torch
import torch.nn as nn
import torch.nn.functional as F

# layer dependencies
from typing import List, Tuple
from timm.models.layers import DropPath, trunc_normal_
from torchvision.utils import _log_api_usage_once

# operation dependencies
import einops
import math

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        linear=False,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = (
                    self.kv(x_)
                    .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
            else:
                kv = (
                    self.kv(x)
                    .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        linear=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        # x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        linear=False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            linear=linear,
        )

        # NOTE: drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            linear=linear,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, batchsize, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        x = x.reshape(batchsize, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


def convx(inchans, outchans, ksize, bias=False):
    return nn.Conv2d(inchans,outchans,ksize,padding=(ksize-1)//2,bias=bias)

def conv1x1(inchans, outchans, bias=False):
    return convx(inchans,outchans,1,bias)

def conv2x2(inchans, outchans, bias=False):
    return convx(inchans,outchans,2,bias)

def conv3x3(inchans, outchans, bias=False):
    return convx(inchans,outchans,3,bias)

class PLVPooling(nn.Module):
    """
    Porpotion of Large Value Pooling Layer
    """

    def __init__(self, channels_first=True, **kwargs):
        super(PLVPooling, self).__init__()
        self.dims = [2, 3] if channels_first else [1, 2]

    def forward(self, x, bias):
        batch, C, H, W = x.shape
        ppvoutput = torch.mean(
            torch.greater(x, torch.reshape(bias, (C, 1, 1)).expand((C, H, W))).float(),
            dim=self.dims,
        )
        return ppvoutput


class FeatureBlock(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels,
        drop_rate
    ):
        super().__init__()
        self.expansion=1

        self.hidden_dims = out_channels*self.expansion
        self.forward_layers = nn.Sequential(
            *[
                nn.Sequential(
                    conv3x3(in_channels if i==0 else self.hidden_dims, self.hidden_dims),
                    nn.BatchNorm2d(self.hidden_dims),
                    nn.ReLU(inplace=True),
                    nn.Dropout(drop_rate)
                )
                for i in range(4)
            ]
        )
        self.post_conv = conv3x3(self.hidden_dims, out_channels, bias=True)
        self.post_act = nn.GELU()

        # for channelwise attention block
        self.pool = PLVPooling()
        
        # output
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.block_drop = nn.Dropout(drop_rate)

    def forward(self, x):
        # forward features
        x = self.forward_layers(x)

        # process for interdeminate result
        x = self.post_conv(x)
        x = self.post_act(x)

        B,C,H,W = x.shape
        # channelwise attention 
        x = torch.multiply(
            self.pool(x, self.post_conv.bias).view(B,C,1,1),
            x
        )

        # block level output
        x = self.downsample(x)
        return self.block_drop(x)

class EncoderWrapper(nn.Module):
    def __init__(self,dim,num_heads,drop) -> None:
        super().__init__()
        self.encoder_layer1 = Encoder(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            drop=drop,
            attn_drop=drop,
            drop_path=drop
        )   
        self.encoder_layer2 = Encoder(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            drop=drop,
            attn_drop=drop,
            drop_path=drop
        )   
    def forward(self,x,B,H,W):
        x = x.flatten(2).transpose(1, 2)
        x = self.encoder_layer1(x,B,H,W)
        x = x.flatten(2).transpose(1, 2)
        x = self.encoder_layer2(x,B,H,W)
        return x

class GDBLS(nn.Module):
    def __init__(
        self,
        num_classes=10,
        input_shape=[3,32,32],
        filters=[128,160,192],
        num_heads=[1,2,4],
        block_drop=0.1,
        overall_drop=0.5
    ):
        super().__init__()

        assert len(filters) == len(num_heads)

        C,H,W = input_shape
        self.block_cnt = len(filters)
        self.num_classes = num_classes

        self.fb_blocks = nn.ModuleList([
            FeatureBlock(
                C if i==0 else filters[i-1],
                filters[i],
                block_drop
            ) for i in range(self.block_cnt)
        ])

        self.local_attn_blocks = nn.ModuleList([
            EncoderWrapper(
                dim=filters[i],
                num_heads=num_heads[i],
                drop=block_drop,
            )   
            for i in range(self.block_cnt)
        ])
        
        # wrappers to resize each output into the shape of the last featureblock output
        self.size_wrapper = nn.ModuleList([
            nn.Sequential(
                *[nn.AvgPool2d(2**(self.block_cnt-1-i)),conv1x1(filters[i],filters[-1])] if i!=self.block_cnt-1 else [nn.Identity()],
                nn.Flatten(start_dim=1),
            ) for i in range(self.block_cnt)
        ])

        self.enhance_block = nn.Dropout(overall_drop)

        wrapped_size = H // (2**self.block_cnt)
        self.cls_head = nn.Linear(
            filters[-1]*wrapped_size*wrapped_size, self.num_classes
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        cur_tensor = x
        ps = []
        for i in range(self.block_cnt):
            # forward each featuremap
            cur_tensor = self.fb_blocks[i](cur_tensor)
            # add local attn
            B,C,H,W = cur_tensor.shape
            cur_tensor = self.local_attn_blocks[i](cur_tensor,B,H,W)
            # align size
            ps.append(self.size_wrapper[i](cur_tensor))

        e_ps = self.enhance_block(sum(ps))
        return self.cls_head(e_ps)
    
if __name__ == "__main__":
    model = GDBLS()
    print(model)
    # Define a function for testing the FeatureBlock module
    def test_feature_block():
        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Create an instance of FeatureBlock
        in_channels = 3  # Replace with your input channels
        out_channels = 64  # Replace with your output channels
        drop_rate = 0.5  # Replace with your desired drop rate
        feature_block = FeatureBlock(in_channels, out_channels, drop_rate)

        # Create a random input tensor
        batch_size = 32  # Replace with your desired batch size
        input_tensor = torch.randn((batch_size, in_channels, 256, 256))  # Adjust input size as needed

        # Forward pass
        output = feature_block(input_tensor)

        # Print the output shape
        print("Output Shape:", output.shape)

        # Perform any additional assertions or checks as needed
        assert output.shape[1] == out_channels  # Ensure the output has the correct number of channels

    # Run the test function
    test_feature_block()
