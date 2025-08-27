import torch.nn as nn 
from torchsummary import summary 

from layers1d import (
    Conv1d_NN, 
    Conv1d_NN_Attn, 
    Conv1d_New
)



class AllConvNet1D(nn.Module): 
    def __init__(self, args): 
        super(AllConvNet1D, self).__init__()
        self.args = args
        self.model = "All Convolutional Network 1D"
        self.name = f"{self.model} {self.args.layer}"
        
        layers = []
        in_ch = self.args.img_size[0] 

        for i in range(self.args.num_layers):
            out_ch = self.args.channels[i]

            # A dictionary to hold parameters for the current layer
            layer_params = {
                "in_channels": in_ch,
                "out_channels": out_ch,
                "shuffle_pattern": self.args.shuffle_pattern,
                "shuffle_scale": self.args.shuffle_scale,
            }

            if self.args.layer == "Conv1d":
                layer = nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=self.args.kernel_size,
                    stride=1,
                    padding='same'
                )
            elif self.args.layer == "Conv1d_New":
                layer = Conv1d_New(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=self.args.kernel_size,
                    stride=1,
                    shuffle_pattern=self.args.shuffle_pattern,
                    shuffle_scale=self.args.shuffle_scale,
                    coordinate_encoding=self.args.coordinate_encoding
                )

            # elif self.args.layer == "Conv2d_New_1d":
            #     layer = Conv2d_New_1d(
            #         in_channels=in_ch, 
            #         out_channels=out_ch, 
            #         K=self.args.K, 
            #         stride=1, 
            #         shuffle_pattern=self.args.shuffle_pattern,
            #         shuffle_scale=self.args.shuffle_scale,
            #         coordinate_encoding=self.args.coordinate_encoding
            #     )

            elif self.args.layer == "ConvNN":
                layer_params.update({
                    "K": self.args.K,
                    "stride": self.args.K, # Stride is always K
                    "sampling_type": self.args.sampling_type,
                    "num_samples": self.args.num_samples,
                    "sample_padding": self.args.sample_padding,
                    "magnitude_type": self.args.magnitude_type,
                    "coordinate_encoding": self.args.coordinate_encoding
                })
                layer = Conv1d_NN(**layer_params)

            elif self.args.layer == "ConvNN_Attn":
                layer_params.update({
                    "K": self.args.K,
                    "stride": self.args.K,
                    "sampling_type": self.args.sampling_type,
                    "num_samples": self.args.num_samples,
                    "sample_padding": self.args.sample_padding,
                    "magnitude_type": self.args.magnitude_type,
                    "num_tokens": self.args.img_size[-1], # Pass H, W
                    "attention_dropout": self.args.attention_dropout,
                    "coordinate_encoding": self.args.coordinate_encoding
                })
                layer = Conv1d_NN_Attn(**layer_params)
            
            # elif "/" in self.args.layer: # Handle all branching cases
                
            #     ch1 = out_ch // 2 if out_ch % 2 == 0 else out_ch // 2 + 1
            #     ch2 = out_ch - ch1
                
            #     layer_params.update({"channel_ratio": (ch1, ch2)})
                
            #     # --- Check all sub-cases for branching layers ---
            #     if self.args.layer == "Conv2d/ConvNN":
            #         layer_params.update({
            #             "kernel_size": self.args.kernel_size,
            #             "K": self.args.K, "stride": self.args.K,
            #             "sampling_type": self.args.sampling_type, "num_samples": self.args.num_samples,
            #             "sample_padding": self.args.sample_padding, "magnitude_type": self.args.magnitude_type,
            #             "coordinate_encoding": self.args.coordinate_encoding
            #         })
            #         layer = Conv2d_ConvNN_Branching(**layer_params)
                
            #     elif self.args.layer == "Conv2d/ConvNN_Attn":
            #         layer_params.update({
            #             "kernel_size": self.args.kernel_size,
            #             "K": self.args.K, "stride": self.args.K,
            #             "sampling_type": self.args.sampling_type, "num_samples": self.args.num_samples,
            #             "sample_padding": self.args.sample_padding, "magnitude_type": self.args.magnitude_type,
            #             "img_size": self.args.img_size[1:],
            #             "coordinate_encoding": self.args.coordinate_encoding
            #         })
            #         layer = Conv2d_ConvNN_Attn_Branching(**layer_params)
                
            #     elif self.args.layer == "Attention/ConvNN":
            #         layer_params.update({
            #             "num_heads": self.args.num_heads,
            #             "K": self.args.K, "stride": self.args.K,
            #             "sampling_type": self.args.sampling_type, "num_samples": self.args.num_samples,
            #             "sample_padding": self.args.sample_padding, "magnitude_type": self.args.magnitude_type,
            #             "coordinate_encoding": self.args.coordinate_encoding
            #         })
            #         layer = Attention_ConvNN_Branching(**layer_params)

            #     elif self.args.layer == "Attention/ConvNN_Attn":
            #         layer_params.update({
            #             "num_heads": self.args.num_heads,
            #             "K": self.args.K, "stride": self.args.K,
            #             "sampling_type": self.args.sampling_type, "num_samples": self.args.num_samples,
            #             "sample_padding": self.args.sample_padding, "magnitude_type": self.args.magnitude_type,
            #             "img_size": self.args.img_size[1:],
            #             "coordinate_encoding": self.args.coordinate_encoding
            #         })
            #         layer = Attention_ConvNN_Attn_Branching(**layer_params)
                
            #     # This is the specific case that was failing
            #     elif self.args.layer == "Conv2d/Attention":
            #         layer_params.update({
            #             "num_heads": self.args.num_heads,
            #             "kernel_size": self.args.kernel_size, 
            #             "coordinate_encoding": self.args.coordinate_encoding
            #         })
            #         layer = Attention_Conv2d_Branching(**layer_params)
                
            #     else:
            #         # This else now only catches unknown branching types
            #         raise ValueError(f"Unknown branching layer type: {self.args.layer}")

            else:
                # This is the final else for non-branching types
                raise ValueError(f"Layer type {self.args.layer} not supported in AllConvNet")

            layers.append(nn.InstanceNorm1d(num_features=out_ch)) # Pre-layer normalization
            layers.append(layer)
            if self.args.layer == "ConvNN_Attn":
                pass #layers.append(nn.Dropout(p=self.args.attention_dropout))
            layers.append(nn.ReLU(inplace=True))
            
            # Update in_ch for the next layer
            in_ch = out_ch
            
        self.features = nn.Sequential(*layers)

        self.output = nn.Conv1d(in_ch, args.img_size[0], kernel_size=1)

        self.to(self.args.device)

    def forward(self, x): 
        x = self.features(x)
        x = self.output(x)
        return x
    
    def summary(self): 
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            # torchsummary expects batch dimension, but img_size doesn't include it
            summary(self, input_size=self.img_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            # Move model back to its original device
            self.to(original_device)
        
    def parameter_count(self): 
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params



if __name__ == "__main__":
    import torch
    from types import SimpleNamespace

    
    base_args = SimpleNamespace(
        layer="Conv1d",
        num_layers=3,
        channels=[4, 4, 4],
        img_size=(1, 40),
        num_classes=10,
        attention_dropout=0.1, 
        # Conv/NN Layer Params
        kernel_size=3,
        K=3,
        magnitude_type="similarity",
        sample_padding=0,
        sampling_type = "spatial", 
        num_samples = 10, 
        # Attention Params
        num_heads=4,
        
        # Shuffle Params
        shuffle_pattern="BA",
        shuffle_scale=2,
        
        
        # Device
        device="mps", 
        coordinate_encoding=True,  # Default to False for simplicity
    )

    x = torch.randn(16, 1, 40).to("mps")
    model = AllConvNet1D(base_args)
    output = model(x)
    print(output.shape)
