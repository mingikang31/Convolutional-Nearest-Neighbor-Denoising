import torch 
import torch.nn as nn 
import torch.nn.functional as F

from models.layers2d import (
    Conv2d_NN, 
    Conv2d_Branching
)


class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__() 


        self.args = args 

        self.in_channels = args.img_size[0]
        self.out_channels = args.img_size[0]
        self.channels = args.channels 
        self.num_pool_layers = args.num_pool_layers

        # Downsampling path
        self.down_layers = nn.ModuleList(
            [
                ConvBlock(args, self.in_channels, self.channels)
            ]
        )
        ch = self.channels
        for _ in range(self.num_pool_layers - 1):
            self.down_layers.append(ConvBlock(args, ch, ch * 2))
            ch *= 2

        # Conv 
        self.conv = ConvBlock(args, ch, ch * 2)

        # Upsampling path
        self.up_layers = nn.ModuleList()
        self.up_transpose_layers = nn.ModuleList()
        for _ in range(self.num_pool_layers - 1):
            self.up_transpose_layers.append(
                TransposeConvBlock(ch*2, ch)
            )
            self.up_layers.append(ConvBlock(args, ch*2, ch))
            ch //= 2

        self.up_transpose_layers.append(
            TransposeConvBlock(ch*2, ch)
        )

        self.up_layers.append(
            nn.Sequential(
                ConvBlock(args, ch*2, ch), 
                nn.Conv2d(ch, self.out_channels, kernel_size=1, stride=1, padding=0)
            )
        )

        self.name = f"UNet {self.args.layer}"

    def forward(self, x):
        # Downsampling path
        skip_connections = [] 
        out = x 
        for layer in self.down_layers:
            out = layer(out)
            skip_connections.append(out)
            out = F.max_pool2d(out, kernel_size=2, stride=2)

        out = self.conv(out) 

        # Upsampling path
        for transpose_layer, up_layer in zip(self.up_transpose_layers, self.up_layers):
            skip = skip_connections.pop()
            out = transpose_layer(out)

            padding = [0, 0, 0, 0] 
            if out.shape[-1] != skip.shape[-1]:
                padding[1] = 1
            if out.shape[-2] != skip.shape[-2]:
                padding[3] = 1
            if torch.sum(torch.tensor(padding)) != 0:
                out = F.pad(out, padding, "constant", 0)

            out = torch.cat((skip, out), dim=1)
            out = up_layer(out)

        return out 

    def parameter_count(self): 
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params



class ConvBlock(nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(ConvBlock, self).__init__() 

        self.args = args 
        self.in_channels = in_channels 
        self.out_channels = out_channels 

        
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

        convnn_branching_params = {
            "kernel_size": self.args.kernel_size,
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
            "branch_ratio": self.args.branch_ratio
        }

        if self.args.layer == "Conv2d":
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        elif self.args.layer == "ConvNN":
            self.conv1 = Conv2d_NN(in_channels, out_channels, **convnn_params)
            self.conv2 = Conv2d_NN(out_channels, out_channels, **convnn_params)
        elif self.args.layer == "Branching":
            self.conv1 = Conv2d_Branching(in_channels, out_channels, **convnn_branching_params)
            self.conv2 = Conv2d_Branching(out_channels, out_channels, **convnn_branching_params)

        self.conv_layer = nn.Sequential(
            self.conv1, 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True), 
            self.conv2,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out = self.conv_layer(x)
        return out



class TransposeConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


if __name__ == "__main__":
    from argparse import Namespace
    args = Namespace()
    args.layer = "Conv2d"
    args.channels = 32
    args.num_pool_layers = 4

    args.K = 9
    args.kernel_size = 3
    args.padding = 1
    args.sampling_type = 'all'
    args.num_samples = -1
    args.sample_padding = 0
    args.shuffle_pattern = "NA"
    args.shuffle_scale = 0
    args.magnitude_type = "cosine"
    args.similarity_type = "Col"
    args.aggregation_type = "Col"
    args.lambda_param = 0.5
    args.branch_ratio = 0.5

    args.img_size = (1, 64, 64)

    model = UNet(args)
    x = torch.randn(2, *args.img_size)
    print(model(x).shape)
    total_params, trainable_params = model.parameter_count()
    print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")