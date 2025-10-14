import torch 
import torch.nn as nn 
import torch.nn.functional as F


from models.layers2d import (
    Conv2d_NN, 
    Conv2d_Branching
)

class DnCNN(nn.Module):
    def __init__(self, args):
        super(DnCNN, self).__init__()

        self.args = args 

        self.in_channels = args.img_size[0]
        self.out_channels = args.img_size[0]
        self.channels = args.channels
        self.num_layers = args.num_layers

        
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


        layers = []

        if self.args.layer == "Conv2d":
            layers.append(nn.Conv2d(in_channels=self.in_channels, out_channels=self.channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))
        elif self.args.layer == "ConvNN":
            layers.append(Conv2d_NN(in_channels=self.in_channels, out_channels=self.channels, **convnn_params))
            layers.append(nn.ReLU(inplace=True))
        elif self.args.layer == "Branching":
            layers.append(Conv2d_Branching(in_channels=self.in_channels, out_channels=self.channels, **convnn_branching_params))
            layers.append(nn.ReLU(inplace=True))

        for _ in range(self.num_layers-2):
            if self.args.layer == "Conv2d":
                layers.append(nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(self.channels))
                layers.append(nn.ReLU(inplace=True))
            elif self.args.layer == "ConvNN":
                layers.append(Conv2d_NN(in_channels=self.channels, out_channels=self.channels, **convnn_params))
                layers.append(nn.BatchNorm2d(self.channels))
                layers.append(nn.ReLU(inplace=True))
            elif self.args.layer == "Branching":
                layers.append(Conv2d_Branching(in_channels=self.channels, out_channels=self.channels, **convnn_branching_params))
                layers.append(nn.BatchNorm2d(self.channels))
                layers.append(nn.ReLU(inplace=True))

        if self.args.layer == "Conv2d":
            layers.append(nn.Conv2d(in_channels=self.channels, out_channels=self.out_channels, kernel_size=3, padding=1, bias=False))
        elif self.args.layer == "ConvNN":
            layers.append(Conv2d_NN(in_channels=self.channels, out_channels=self.out_channels, **convnn_params))
        elif self.args.layer == "Branching":
            layers.append(Conv2d_Branching(in_channels=self.channels, out_channels=self.out_channels, **convnn_branching_params))

        self.dncnn = nn.Sequential(*layers)

        self.name = f"DnCNN {self.args.layer}"

    def forward(self, x):
        out = self.dncnn(x)
        return out
    
    def parameter_count(self): 
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

        
if __name__ == "__main__":
    from argparse import Namespace
    args = Namespace()
    args.layer = "ConvNN"
    args.channels = 32
    args.num_layers = 4

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

    args.img_size = (3, 64, 64)

    model = DnCNN(args)
    x = torch.randn(2, *args.img_size)
    print(model(x).shape)
    total_params, trainable_params = model.parameter_count()
    print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")