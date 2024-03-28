import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_HOME'] = '/media/stu/74A84304A842C478/ConvNextUnet/pretrained_ckpt'
import torch.nn as nn
import torch
from network.soft_attention import Attention_block
import torch.nn.functional as F
class Inception_Block(nn.Module):
    def __init__(self, in_channels=1, n_feat_first_layer=[12, 12, 12]):
        super(Inception_Block, self).__init__()
        self.conv_3 = nn.Conv2d(in_channels, n_feat_first_layer[0], kernel_size=3, padding=1)
        self.conv_5 = nn.Conv2d(in_channels, n_feat_first_layer[1], kernel_size=5, padding=2)
        self.conv_7 = nn.Conv2d(in_channels, n_feat_first_layer[2], kernel_size=7, padding=3)
        self.BN = nn.BatchNorm2d(n_feat_first_layer[0] + n_feat_first_layer[1] + n_feat_first_layer[2])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.conv_3(x)
        out2 = self.conv_5(x)
        out3 = self.conv_7(x)
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.BN(out)
        out = self.relu(out)
        return out


class LayerNorm(nn.Module):
    """
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        # self.bn = BatchNorm2d(out_chan)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

from timm.models.layers import DropPath
class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class UpConvNext(nn.Module):
    def __init__(self, in_channels, out_channels, block_num=2):
        super(UpConvNext, self).__init__()
        self.upscale_factor = 2

        self.norm = LayerNorm(in_channels, eps=1e-6)
        self.up = nn.Upsample(scale_factor=self.upscale_factor, mode='bilinear')
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        drop_path_rate = 0.3
        depths = [3, 3, 9, 3]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.block_num = block_num

        self.block1 = nn.Sequential(
            Block(dim=out_channels, drop_path=dp_rates[-4], layer_scale_init_value=1e-6),
            Block(dim=out_channels, drop_path=dp_rates[-5], layer_scale_init_value=1e-6),
            Block(dim=out_channels, drop_path=dp_rates[-6], layer_scale_init_value=1e-6),
            Block(dim=out_channels, drop_path=dp_rates[-7], layer_scale_init_value=1e-6),
            Block(dim=out_channels, drop_path=dp_rates[-8], layer_scale_init_value=1e-6),
            Block(dim=out_channels, drop_path=dp_rates[-9], layer_scale_init_value=1e-6),
            Block(dim=out_channels, drop_path=dp_rates[-10], layer_scale_init_value=1e-6),
            Block(dim=out_channels, drop_path=dp_rates[-11], layer_scale_init_value=1e-6),
            Block(dim=out_channels, drop_path=dp_rates[-12], layer_scale_init_value=1e-6),
        )
        self.block2 = nn.Sequential(
            Block(dim=out_channels, drop_path=dp_rates[-13], layer_scale_init_value=1e-6),
            Block(dim=out_channels, drop_path=dp_rates[-14], layer_scale_init_value=1e-6),
            Block(dim=out_channels, drop_path=dp_rates[-15], layer_scale_init_value=1e-6),
        )
        self.block3 = nn.Sequential(
            Block(dim=out_channels, drop_path=dp_rates[-16], layer_scale_init_value=1e-6),
            Block(dim=out_channels, drop_path=dp_rates[-17], layer_scale_init_value=1e-6),
            Block(dim=out_channels, drop_path=dp_rates[-18], layer_scale_init_value=1e-6),
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.up(x)
        x = self.conv(x)

        if self.block_num == 2:
            x = self.block1(x)
        elif self.block_num == 1:
            x = self.block2(x)
        elif self.block_num == 0:
            x = self.block3(x)
        elif self.block_num == 3:
            x = x
        return x


class Up_conv(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(Up_conv, self).__init__()
        self.con = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm = LayerNorm(mid_channels, eps=1e-6)
        self.gelu = nn.GELU()
        self.con2 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = self.con(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.gelu(x)
        x = self.con2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.gelu(x)
        return x


class same_Up_conv(nn.Module):
    def __init__(self, mid_channels):
        super(same_Up_conv, self).__init__()
        self.con1 = nn.ConvTranspose2d(192, mid_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm = LayerNorm(mid_channels, eps=1e-6)
        self.gelu = nn.GELU()
        self.con2 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = self.con1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.gelu(x)
        x = self.con2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.gelu(x)
        return x


class Attention_UP(nn.Module):
    def __init__(self, in_channels=128, mid_channels=64, out_channels=32, block_idx=2):
        super(Attention_UP, self).__init__()
        if block_idx == 2:
            self.Up = UpConvNext(in_channels, mid_channels, 2)
        elif block_idx == 1:
            self.Up = UpConvNext(in_channels, mid_channels, 1)
        elif block_idx == 0:
            self.Up = UpConvNext(in_channels, mid_channels, 0)
        elif block_idx == 3:
            self.Up = UpConvNext(in_channels, mid_channels, 3)
        self.Atten = Attention_block(mid_channels, mid_channels, out_channels)
        self.Atten1 = Attention_block(in_channels, mid_channels, out_channels)
        self.Up_conv = Up_conv(in_channels, mid_channels)
        self.same_Up_conv = same_Up_conv(mid_channels)

    def forward(self, x1, x2):
        if x2.shape[2] == 56:
            d2 = x2
            x1 = self.Atten1(g=d2, x=x1)
            d2 = torch.cat((x1, d2), dim=1)
            d2 = self.same_Up_conv(d2)
        else:
            d2 = self.Up(x2)
            x1 = self.Atten(g=d2, x=x1)
            d2 = torch.cat((x1, d2), dim=1)
            d2 = self.Up_conv(d2)
        return d2, x1


from torchvision import models
class ConvNextUNet(nn.Module):
    def __init__(self, num_classes, in_channels=1, dims=[96, 192, 384, 768]):
        super(ConvNextUNet, self).__init__()
        self.x00_down = Inception_Block(1, [16, 16, 16])  # channels=36  [4, 4, 4]  224 64
        ConvNeXt_T = models.convnext_tiny(pretrained=True)

        # self.first = ConvNeXt_T.features[0]
        self.first = nn.Sequential(
            nn.Conv2d(48, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        # self.detail = BiSeNetOutput(48, 24, 1)
        self.encoder1 = ConvNeXt_T.features[1]  # 96
        self.detail1 = BiSeNetOutput(96, 48, 1)  # 96,64,1
        self.encoder2 = ConvNeXt_T.features[2:4]  # 192
        self.detail2 = BiSeNetOutput(192, 96, 1)
        self.encoder3 = ConvNeXt_T.features[4:6]  # 384
        self.encoder4 = ConvNeXt_T.features[6:8]  # 768

        # column2
        self.x31_up = Attention_UP(768, 384, 192, 2)
        self.x21_up = Attention_UP(384, 192, 96, 1)
        self.x11_up = Attention_UP(192, 96, 96, 0)
        self.x01_up = Attention_UP(96, 96, 64, 3)

        self.finalconv1 = nn.ConvTranspose2d(96, 48, kernel_size=4, stride=4)  # 4倍上采样
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalbn1 = nn.BatchNorm2d(48)  ##
        self.finalconv2 = nn.Conv2d(48, 48, kernel_size=2)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.fibalbn2 = nn.BatchNorm2d(48)  ##

        # self.classification = nn.Conv2d(32, num_classes, kernel_size=1)
        self.classification = nn.Conv2d(48, num_classes, kernel_size=2, padding=1)

    def forward(self, x):
        # x = self.x00_down1(x)
        x00 = self.x00_down(x)
        # x00 = self.detail(x00)
        x00 = self.first(x00)
        x10 = self.encoder1(x00)
        # x10 = self.detail1(x10)
        x20 = self.encoder2(x10)
        # x20 = self.detail2(x20)
        x30 = self.encoder3(x20)
        x40 = self.encoder4(x30)

        x31, x30 = self.x31_up(x30, x40)
        x21, x20 = self.x21_up(x20, x30)
        x11, x10 = self.x11_up(x10, x21)
        x01, x00 = self.x01_up(x00, x11)

        x = self.finalconv1(x01)
        x = self.finalrelu1(x)
        # x = self.finalbn1(x)  ##
        x = self.finalconv2(x)
        x = self.finalrelu2(x)
        # x = self.fibalbn2(x)  ##
        x = self.classification(x)
        return x


if __name__ == '__main__':
    from torchsummary import summary

    num_classes = 4
    image = torch.randn(1, 1, 224, 224).cuda()
    model = ConvNextUNet(num_classes).cuda()
    model.eval()
    with torch.no_grad():
        output = model.forward(image)
    print(output.size())