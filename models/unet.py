import math
from typing import List, Tuple
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F

from models.layers2d import (
    Conv2d_New, 
    Conv2d_NN, 
    Conv2d_NN_Attn
)

class Unet(nn.Module):

    def __init__(
        self,
        args
        # in_chans: int,
        # out_chans: int,
        # chans: int = 32,
        # num_pool_layers: int = 4,
        # drop_prob: float = 0.0,
    ):
       
        super().__init__()

        self.args = args

        self.in_chans = args.img_size[0]
        self.out_chans = args.img_size[0]
        self.chans = args.chans
        self.num_pool_layers = args.num_pool_layers
        self.drop_prob = args.drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(args, self.in_chans, self.chans, self.drop_prob)])
        ch = self.chans
        for _ in range(self.num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(args, ch, ch * 2, self.drop_prob))
            ch *= 2
        self.conv = ConvBlock(args, ch, ch * 2, self.drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(self.num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(self.args, ch * 2, ch, self.drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(args,ch * 2, ch, self.drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

        self.name = f"UNet {self.args.layer}"

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        stack = []
        output = image

        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output
   
    def parameter_count(self): 
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


class ConvBlock(nn.Module):
    def __init__(self, args, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()
        self.args = args
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        conv2d_new_params = {
            "kernel_size": self.args.kernel_size,
            "stride": 1, # Stride is always 1 
            "shuffle_pattern": self.args.shuffle_pattern,
            "shuffle_scale": self.args.shuffle_scale,
            "aggregation_type": self.args.aggregation_type
        }

        convnn_params = {
            "K": self.args.K, 
            "stride": self.args.K, # Stride is always K
            "padding": self.args.padding,
            "sampling_type": self.args.sampling_type,
            "num_samples": self.args.num_samples,
            "sample_padding": self.args.sample_padding,
            "shuffle_pattern": self.args.shuffle_pattern,
            "shuffle_scale": self.args.shuffle_scale,
            "magnitude_type": self.args.magnitude_type,
            "similarity_type": self.args.similarity_type,
            "aggregation_type": self.args.aggregation_type, 
            "lambda_param": self.args.lambda_param
        }
        
        convnn_attn_params = {
            "K": self.args.K, 
            "stride": self.args.K, # Stride is always K
            "padding": self.args.padding,
            "sampling_type": self.args.sampling_type,
            "num_samples": self.args.num_samples,
            "sample_padding": self.args.sample_padding,
            "shuffle_pattern": self.args.shuffle_pattern,
            "shuffle_scale": self.args.shuffle_scale,
            "magnitude_type": self.args.magnitude_type,
            "similarity_type": self.args.similarity_type,
            "aggregation_type": self.args.aggregation_type, 
            "lambda_param": self.args.lambda_param,
            
            "attention_dropout": self.args.attention_dropout
        }
        
        if self.args.layer == "Conv2d":
            self.layers = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(out_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Dropout2d(drop_prob),
                nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(out_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Dropout2d(drop_prob),
            )
        elif self.args.layer == "ConvNN":
            
            
            self.layers = nn.Sequential(
                Conv2d_NN(in_chans, out_chans, **convnn_params),
                nn.InstanceNorm2d(out_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Dropout2d(drop_prob),
                Conv2d_NN(out_chans, out_chans, **convnn_params),
                nn.InstanceNorm2d(out_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Dropout2d(drop_prob),
            )
        elif self.args.layer == "ConvNN_Attn":
            
            self.layers = nn.Sequential(
                Conv2d_NN_Attn(in_chans, out_chans, **convnn_attn_params),
                nn.InstanceNorm2d(out_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Dropout2d(drop_prob),
                Conv2d_NN_Attn(out_chans, out_chans, **convnn_attn_params),
                nn.InstanceNorm2d(out_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Dropout2d(drop_prob),
            )
            

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)

if __name__ == "__main__":  
     # Create default args
    args = SimpleNamespace(
        layer="ConvNN",
        num_pool_layers=3, 
        chans=32, 
        drop_prob=0.1,
        K=9,
        sampling_type="all",
        num_samples=-1,
        sample_padding=0,
        num_heads=4,
        attention_dropout=0.1,
        shuffle_pattern="NA",
        shuffle_scale=2,
        magnitude_type="similarity",
        coordinate_encoding=False, 
        img_size=(3, 112, 112), 
        num_classes=10,
    )
    model = Unet(
        args=args
    )
    print("Parameter count ConvNN: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    ex = torch.randn(1, 3, 32, 32)
    out = model(ex)
    print("Output shape ConvNN: ", out.shape)

    args.layer = "Conv2d"
    model = Unet(
        args=args
    )
    print("Parameter count Conv2d: ", sum(p.numel() for p in model.parameters() if p.requires_grad))