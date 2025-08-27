"""Convolutional Nearest Neighbor Layers 2D"""

"""
Layers 2D: 
(1) Conv2d_NN (All, Random, Spatial Sampling) 
(2) Conv2d_NN_Attn (All, Random, Spatial Sampling) 
(3) Attention2d 

Branching Layers 2D: 
(4) Conv2d_ConvNN_Branching (All, Random, Spatial Sampling)
(5) Conv2d_ConvNN_Attn_Branching (All, Random, Spatial Sampling)
(6) Attention_ConvNN_Branching (All, Random, Spatial Sampling)
(7) Attention_ConvNN_Attn_Branching (All, Random, Spatial Sampling)
(8) Attention_Conv2d_Branching 
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
# from models.layers1d import Attention1d

class Conv2d_New(nn.Module): 
    """Convolution 2D Nearest Neighbor Layer"""
    def __init__(self, 
                in_channels, 
                out_channels, 
                kernel_size,
                stride, 
                shuffle_pattern, 
                shuffle_scale, 
                coordinate_encoding
                ): 
        
        super(Conv2d_New, self).__init__()
        
        # Assertions 
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        
        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale

        # Positional Encoding (optional)
        self.coordinate_encoding = coordinate_encoding
        self.coordinate_cache = {} 
        self.in_channels = in_channels + 2 if self.coordinate_encoding else in_channels
        self.out_channels = out_channels + 2 if self.coordinate_encoding else out_channels

        # Shuffle2D/Unshuffle2D Layers
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)
        
        # Adjust Channels for PixelShuffle
        self.in_channels_shuff = self.in_channels * (self.shuffle_scale**2) if self.shuffle_pattern in ["B", "BA"] else self.in_channels
        self.out_channels_shuff = self.out_channels * (self.shuffle_scale**2) if self.shuffle_pattern in ["A", "BA"] else self.out_channels

        # Conv2d Layer
        self.conv2d_layer = nn.Conv2d(in_channels=self.in_channels_shuff, 
                                      out_channels=self.out_channels_shuff, 
                                      kernel_size=self.kernel_size, 
                                      stride=self.stride, 
                                      padding="same")


        # Pointwise Convolution Layer
        self.pointwise_conv = nn.Conv2d(in_channels=self.out_channels,
                                         out_channels=self.out_channels - 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        
        
    def forward(self, x): 
        # Coordinate Channels (optional) + Unshuffle + Flatten 
        x = self._add_coordinate_encoding(x) if self.coordinate_encoding else x
        x_2d = self.unshuffle_layer(x) if self.shuffle_pattern in ["B", "BA"] else x

        # Conv2d Layer
        x = self.conv2d_layer(x_2d)

        x = self.shuffle_layer(x) if self.shuffle_pattern in ["A", "BA"] else x
        x = self.pointwise_conv(x) if self.coordinate_encoding else x
        return x

    def _add_coordinate_encoding(self, x):
        b, _, h, w = x.shape
        cache_key = f"{b}_{h}_{w}_{x.device}"

        if cache_key in self.coordinate_cache:
            expanded_grid = self.coordinate_cache[cache_key]
        else:
            y_coords_vec = torch.linspace(start=-1, end=1, steps=h, device=x.device)
            x_coords_vec = torch.linspace(start=-1, end=1, steps=w, device=x.device)

            y_grid, x_grid = torch.meshgrid(y_coords_vec, x_coords_vec, indexing='ij')
            grid = torch.stack((x_grid, y_grid), dim=0).unsqueeze(0)
            expanded_grid = grid.expand(b, -1, -1, -1)
            self.coordinate_cache[cache_key] = expanded_grid

        x_with_coords = torch.cat((x, expanded_grid), dim=1)
        return x_with_coords
    
class Conv2d_New_1d(nn.Module): 
    """Convolution 2D Nearest Neighbor Layer"""
    def __init__(self, 
                in_channels, 
                out_channels, 
                K,
                stride, 
                shuffle_pattern, 
                shuffle_scale, 
                coordinate_encoding
                ): 
        """
        Parameters: 
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            K (int): Number of Nearest Neighbors for consideration.
            stride (int): Stride size.
            sampling_type (str): Sampling type: "all", "random", "spatial".
            num_samples (int): Number of samples to consider. -1 for all samples.
            shuffle_pattern (str): Shuffle pattern: "B", "A", "BA".
            shuffle_scale (int): Shuffle scale factor.
            magnitude_type (str): Distance or Similarity.
        """
        super(Conv2d_New_1d, self).__init__()
        
        # Assertions 
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        
        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale

        # Positional Encoding (optional)
        self.coordinate_encoding = coordinate_encoding
        self.coordinate_cache = {} 
        self.in_channels = in_channels + 2 if self.coordinate_encoding else in_channels
        self.out_channels = out_channels + 2 if self.coordinate_encoding else out_channels

        # Shuffle2D/Unshuffle2D Layers
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)
        
        # Adjust Channels for PixelShuffle
        self.in_channels_1d = self.in_channels * (self.shuffle_scale**2) if self.shuffle_pattern in ["B", "BA"] else self.in_channels
        self.out_channels_1d = self.out_channels * (self.shuffle_scale**2) if self.shuffle_pattern in ["A", "BA"] else self.out_channels

        # Conv1d Layer
        self.conv1d_layer = nn.Conv1d(in_channels=self.in_channels_1d, 
                                      out_channels=self.out_channels_1d, 
                                      kernel_size=self.K, 
                                      stride=self.stride, 
                                      padding='same')

        # Flatten Layer
        self.flatten = nn.Flatten(start_dim=2)

        # Pointwise Convolution Layer
        self.pointwise_conv = nn.Conv2d(in_channels=self.out_channels,
                                         out_channels=self.out_channels - 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        
        

    def forward(self, x): 
        # Coordinate Channels (optional) + Unshuffle + Flatten 
        x = self._add_coordinate_encoding(x) if self.coordinate_encoding else x
        x_2d = self.unshuffle_layer(x) if self.shuffle_pattern in ["B", "BA"] else x
        x = self.flatten(x_2d)

        # Post-Processing 
        x_conv = self.conv1d_layer(x) 

        # Unflatten + Shuffle
        unflatten = nn.Unflatten(dim=2, unflattened_size=x_2d.shape[2:])
        x = unflatten(x_conv)  # [batch_size, out_channels
        x = self.shuffle_layer(x) if self.shuffle_pattern in ["A", "BA"] else x
        x = self.pointwise_conv(x) if self.coordinate_encoding else x
        return x

    def _add_coordinate_encoding(self, x):
        b, _, h, w = x.shape
        cache_key = f"{b}_{h}_{w}_{x.device}"

        if cache_key in self.coordinate_cache:
            expanded_grid = self.coordinate_cache[cache_key]
        else:
            y_coords_vec = torch.linspace(start=-1, end=1, steps=h, device=x.device)
            x_coords_vec = torch.linspace(start=-1, end=1, steps=w, device=x.device)

            y_grid, x_grid = torch.meshgrid(y_coords_vec, x_coords_vec, indexing='ij')
            grid = torch.stack((x_grid, y_grid), dim=0).unsqueeze(0)
            expanded_grid = grid.expand(b, -1, -1, -1)
            self.coordinate_cache[cache_key] = expanded_grid

        x_with_coords = torch.cat((x, expanded_grid), dim=1)
        return x_with_coords
    
"""(1) Conv2d_NN (All, Random, Spatial Sampling)"""
class Conv2d_NN(nn.Module): 
    """Convolution 2D Nearest Neighbor Layer"""
    def __init__(self, 
                in_channels, 
                out_channels, 
                K,
                stride, 
                sampling_type, 
                num_samples, 
                sample_padding,
                shuffle_pattern, 
                shuffle_scale, 
                magnitude_type,
                coordinate_encoding
                ): 
        """
        Parameters: 
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            K (int): Number of Nearest Neighbors for consideration.
            stride (int): Stride size.
            sampling_type (str): Sampling type: "all", "random", "spatial".
            num_samples (int): Number of samples to consider. -1 for all samples.
            shuffle_pattern (str): Shuffle pattern: "B", "A", "BA".
            shuffle_scale (int): Shuffle scale factor.
            magnitude_type (str): Distance or Similarity.
        """
        super(Conv2d_NN, self).__init__()
        
        # Assertions 
        assert K == stride, "Error: K must be same as stride. K == stride."
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert sampling_type in ["all", "random", "spatial"], "Error: sampling_type must be one of ['all', 'random', 'spatial']"
        assert int(num_samples) > 0 or int(num_samples) == -1, "Error: num_samples must be greater than 0 or -1 for all samples"
        assert (sampling_type == "all" and int(num_samples) == -1) or (sampling_type != "all" and isinstance(num_samples, int)), "Error: num_samples must be -1 for 'all' sampling or an integer for 'random' and 'spatial' sampling"
        
        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride
        self.sampling_type = sampling_type
        self.num_samples = num_samples if num_samples != -1 else 'all'  # -1 for all samples
        self.sample_padding = sample_padding if sampling_type == "spatial" else 0
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.magnitude_type = magnitude_type
        self.maximum = True if self.magnitude_type == 'similarity' else False
        self.INF_DISTANCE = 1e10
        self.NEG_INF_DISTANCE = -1e10

        # Positional Encoding (optional)
        self.coordinate_encoding = coordinate_encoding
        self.coordinate_cache = {} 
        self.in_channels = in_channels + 2 if self.coordinate_encoding else in_channels
        self.out_channels = out_channels + 2 if self.coordinate_encoding else out_channels

        # Shuffle2D/Unshuffle2D Layers
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)
        
        # Adjust Channels for PixelShuffle
        self.in_channels_1d = self.in_channels * (self.shuffle_scale**2) if self.shuffle_pattern in ["B", "BA"] else self.in_channels
        self.out_channels_1d = self.out_channels * (self.shuffle_scale**2) if self.shuffle_pattern in ["A", "BA"] else self.out_channels

        # Conv1d Layer
        self.conv1d_layer = nn.Conv1d(in_channels=self.in_channels_1d, 
                                      out_channels=self.out_channels_1d, 
                                      kernel_size=self.K, 
                                      stride=self.stride, 
                                      padding=0)

        # Flatten Layer
        self.flatten = nn.Flatten(start_dim=2)

        # Pointwise Convolution Layer
        self.pointwise_conv = nn.Conv2d(in_channels=self.out_channels,
                                         out_channels=self.out_channels - 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        
        

    def forward(self, x): 
        # Coordinate Channels (optional) + Unshuffle + Flatten 
        x = self._add_coordinate_encoding(x) if self.coordinate_encoding else x
        x_2d = self.unshuffle_layer(x) if self.shuffle_pattern in ["B", "BA"] else x
        x = self.flatten(x_2d)

        if self.sampling_type == "all":    
            # ConvNN Algorithm 
            matrix_magnitude = self._calculate_distance_matrix(x, sqrt=True) if self.magnitude_type == 'distance' else self._calculate_similarity_matrix(x)
            prime = self._prime(x, matrix_magnitude, self.K, self.maximum)
             
        elif self.sampling_type == "random":
            # Select random samples
            rand_idx = torch.randperm(x.shape[2], device=x.device)[:self.num_samples]
            x_sample = x[:, :, rand_idx]

            # ConvNN Algorithm 
            matrix_magnitude = self._calculate_distance_matrix_N(x, x_sample, sqrt=True) if self.magnitude_type == 'distance' else self._calculate_similarity_matrix_N(x, x_sample)
            range_idx = torch.arange(len(rand_idx), device=x.device)
            matrix_magnitude[:, rand_idx, range_idx] = self.INF_DISTANCE if self.magnitude_type == 'distance' else self.NEG_INF_DISTANCE
            prime = self._prime_N(x, matrix_magnitude, self.K, rand_idx, self.maximum)
            
        elif self.sampling_type == "spatial":
            # Get spatial sampled indices
            x_ind = torch.linspace(0 + self.sample_padding, x_2d.shape[2] - self.sample_padding - 1, self.num_samples, device=x.device).to(torch.long)
            y_ind = torch.linspace(0 + self.sample_padding, x_2d.shape[3] - self.sample_padding - 1, self.num_samples, device=x.device).to(torch.long)
            x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
            x_idx_flat, y_idx_flat = x_grid.flatten(), y_grid.flatten()
            width = x_2d.shape[2] 
            flat_indices = y_idx_flat * width + x_idx_flat  
            x_sample = x[:, :, flat_indices]

            # ConvNN Algorithm
            matrix_magnitude = self._calculate_distance_matrix_N(x, x_sample, sqrt=True) if self.magnitude_type == 'distance' else self._calculate_similarity_matrix_N(x, x_sample)
            range_idx = torch.arange(len(flat_indices), device=x.device)
            matrix_magnitude[:, flat_indices, range_idx] = self.INF_DISTANCE if self.magnitude_type == 'distance' else self.NEG_INF_DISTANCE
            prime = self._prime_N(x, matrix_magnitude, self.K, flat_indices, self.maximum)
        else: 
            raise ValueError("Invalid sampling_type. Must be one of ['all', 'random', 'spatial'].")

        # Post-Processing 
        x_conv = self.conv1d_layer(prime) 
        
        # Unflatten + Shuffle
        unflatten = nn.Unflatten(dim=2, unflattened_size=x_2d.shape[2:])
        x = unflatten(x_conv)  # [batch_size, out_channels
        x = self.shuffle_layer(x) if self.shuffle_pattern in ["A", "BA"] else x
        x = self.pointwise_conv(x) if self.coordinate_encoding else x
        return x

    def _calculate_distance_matrix(self, matrix, sqrt=False):
        norm_squared = torch.sum(matrix ** 2, dim=1, keepdim=True)
        dot_product = torch.bmm(matrix.transpose(2, 1), matrix)
        
        dist_matrix = norm_squared + norm_squared.transpose(2, 1) - 2 * dot_product
        dist_matrix = torch.clamp(dist_matrix, min=-1.0, max=1.0) # remove negative values
        dist_matrix = torch.sqrt(dist_matrix) if sqrt else dist_matrix # take square root if needed
        
        return dist_matrix
    
    def _calculate_distance_matrix_N(self, matrix, matrix_sample, sqrt=False):
        norm_squared = torch.sum(matrix ** 2, dim=1, keepdim=True).permute(0, 2, 1)
        norm_squared_sample = torch.sum(matrix_sample ** 2, dim=1, keepdim=True).transpose(2, 1).permute(0, 2, 1)
        dot_product = torch.bmm(matrix.transpose(2, 1), matrix_sample)
        
        dist_matrix = norm_squared + norm_squared_sample - 2 * dot_product
        dist_matrix = torch.clamp(dist_matrix, min=-1.0, max=1.0) 
        dist_matrix = torch.sqrt(dist_matrix) if sqrt else dist_matrix

        return dist_matrix
    
    def _calculate_similarity_matrix(self, matrix):
        # p=2 (L2 Norm - Euclidean Distance), dim=1 (across the channels)
        norm_matrix = F.normalize(matrix, p=2, dim=1) 
        similarity_matrix = torch.bmm(norm_matrix.transpose(2, 1), norm_matrix)
        similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0) 
        return similarity_matrix
    
    def _calculate_similarity_matrix_N(self, matrix, matrix_sample):
        # p=2 (L2 Norm - Euclidean Distance), dim=1 (across the channels)
        norm_matrix = F.normalize(matrix, p=2, dim=1) 
        norm_sample = F.normalize(matrix_sample, p=2, dim=1)
        similarity_matrix = torch.bmm(norm_matrix.transpose(2, 1), norm_sample)
        similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0) 
        return similarity_matrix

    # def _prime(self, matrix, magnitude_matrix, K, maximum):
    #     b, c, t = matrix.shape
    #     _, topk_indices = torch.topk(magnitude_matrix, k=K, dim=2, largest=maximum)
    #     topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)    
      
    #     matrix_expanded = matrix.unsqueeze(-1).expand(b, c, t, K).contiguous()
    #     prime = torch.gather(matrix_expanded, dim=2, index=topk_indices_exp)
    #     prime = prime.view(b, c, -1)
    #     return prime
    
    def _prime(self, matrix, magnitude_matrix, K, maximum):
        b, c, t = matrix.shape
        topk_values, topk_indices = torch.topk(magnitude_matrix, k=K, dim=2, largest=maximum)
        topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)    
        topk_values_exp = topk_values.unsqueeze(1).expand(b, c, t, K)    
        matrix_expanded = matrix.unsqueeze(-1).expand(b, c, t, K).contiguous()
        prime = torch.gather(matrix_expanded, dim=2, index=topk_indices_exp)
        prime = topk_values_exp * prime
        prime = prime.view(b, c, -1)
        return prime
    
    # def _prime_N_old(self, matrix, magnitude_matrix, K, rand_idx, maximum):
    #     b, c, t = matrix.shape
    #     _, topk_indices = torch.topk(magnitude_matrix, k=K - 1, dim=2, largest=maximum)
    #     tk = topk_indices.shape[-1]
    #     assert K == tk + 1, "Error: K must be same as tk + 1. K == tk + 1."

    #     mapped_tensor = rand_idx[topk_indices]
    #     token_indices = torch.arange(t, device=matrix.device).view(1, t, 1).expand(b, t, 1)
    #     final_indices = torch.cat([token_indices, mapped_tensor], dim=2)
    #     indices_expanded = final_indices.unsqueeze(1).expand(b, c, t, K)

    #     matrix_expanded = matrix.unsqueeze(-1).expand(b, c, t, K).contiguous()
    #     prime = torch.gather(matrix_expanded, dim=2, index=indices_expanded)  
    #     prime = prime.view(b, c, -1)
    #     return prime
    
    def _prime_N(self, matrix, magnitude_matrix, K, rand_idx, maximum):
        b, c, t = matrix.shape
        
        # Get both topk_values AND topk_indices (was missing topk_values before)
        topk_values, topk_indices = torch.topk(magnitude_matrix, k=K - 1, dim=2, largest=maximum)
        tk = topk_indices.shape[-1]
        assert K == tk + 1, "Error: K must be same as tk + 1. K == tk + 1."

        # Map sample indices back to original matrix positions
        mapped_tensor = rand_idx[topk_indices]
        token_indices = torch.arange(t, device=matrix.device).view(1, t, 1).expand(b, t, 1)
        final_indices = torch.cat([token_indices, mapped_tensor], dim=2)
        indices_expanded = final_indices.unsqueeze(1).expand(b, c, t, K)

        # Create similarity weights: [1.0 for self, topk_values for neighbors]
        topk_values_exp = topk_values.unsqueeze(1).expand(b, c, t, K-1)  # Expand for K-1 neighbors
        ones = torch.ones((b, c, t, 1), device=matrix.device)            # Weight=1.0 for self
        topk_values_exp = torch.cat((ones, topk_values_exp), dim=-1)     # Combine: [self=1.0, neighbors=similarities]

        # Gather matrix values and apply similarity weighting
        matrix_expanded = matrix.unsqueeze(-1).expand(b, c, t, K).contiguous()
        prime = torch.gather(matrix_expanded, dim=2, index=indices_expanded)  
        prime = topk_values_exp * prime  # 🔥 KEY ADDITION: Apply similarity weighting
        prime = prime.view(b, c, -1)
        return prime
    
    def _add_coordinate_encoding(self, x):
        b, _, h, w = x.shape
        cache_key = f"{b}_{h}_{w}_{x.device}"

        if cache_key in self.coordinate_cache:
            expanded_grid = self.coordinate_cache[cache_key]
        else:
            y_coords_vec = torch.linspace(start=-1, end=1, steps=h, device=x.device)
            x_coords_vec = torch.linspace(start=-1, end=1, steps=w, device=x.device)

            y_grid, x_grid = torch.meshgrid(y_coords_vec, x_coords_vec, indexing='ij')
            grid = torch.stack((x_grid, y_grid), dim=0).unsqueeze(0)
            expanded_grid = grid.expand(b, -1, -1, -1)
            self.coordinate_cache[cache_key] = expanded_grid

        x_with_coords = torch.cat((x, expanded_grid), dim=1)
        return x_with_coords

"""(2) Conv2d_NN_Attn (All, Random, Spatial Sampling)"""
class Conv2d_NN_Attn(nn.Module): 
    """Convolution 2D Nearest Neighbor Layer"""
    def __init__(self, 
                in_channels, 
                out_channels, 
                K,
                stride, 
                sampling_type, 
                num_samples, 
                sample_padding,
                shuffle_pattern, 
                shuffle_scale, 
                magnitude_type,
                img_size, 
                attention_dropout,
                coordinate_encoding
                ): 
        """
        Parameters: 
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            K (int): Number of Nearest Neighbors for consideration.
            stride (int): Stride size.
            sampling_type (str): Sampling type: "all", "random", "spatial".
            num_samples (int): Number of samples to consider. -1 for all samples.
            shuffle_pattern (str): Shuffle pattern: "B", "A", "BA".
            shuffle_scale (int): Shuffle scale factor.
            magnitude_type (str): Distance or Similarity.
            img_size (tuple): Size of the input image (height, width) for attention.
        """
        super(Conv2d_NN_Attn, self).__init__()
        
        # Assertions 
        assert K == stride, "Error: K must be same as stride. K == stride."
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert sampling_type in ["all", "random", "spatial"], "Error: sampling_type must be one of ['all', 'random', 'spatial']"
        assert int(num_samples) > 0 or int(num_samples) == -1, "Error: num_samples must be greater than 0 or -1 for all samples"
        assert (sampling_type == "all" and int(num_samples) == -1) or (sampling_type != "all" and isinstance(num_samples, int)), "Error: num_samples must be -1 for 'all' sampling or an integer for 'random' and 'spatial' sampling"
        
        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride
        self.sampling_type = sampling_type
        self.num_samples = num_samples if num_samples != -1 else 'all'  # -1 for all samples
        self.sample_padding = sample_padding if sampling_type == "spatial" else 0
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.magnitude_type = magnitude_type
        self.maximum = True if self.magnitude_type == 'similarity' else False
        self.INF_DISTANCE = 1e10 
        self.NEG_INF_DISTANCE = -1e10

        self.img_size = img_size  # Image size for spatial sampling
        self.num_tokens = int((img_size[0] * img_size[1]) / (shuffle_scale**2)) if self.shuffle_pattern in ["B", "BA"] else (img_size[0] * img_size[1])

        # Positional Encoding (optional)
        self.coordinate_encoding = coordinate_encoding
        self.coordinate_cache = {} 
        self.in_channels = in_channels + 2 if self.coordinate_encoding else in_channels
        self.out_channels = out_channels + 2 if self.coordinate_encoding else out_channels
        
        # Shuffle2D/Unshuffle2D Layers
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)
        
        # Adjust Channels for PixelShuffle
        self.in_channels_1d = self.in_channels * (self.shuffle_scale**2) if self.shuffle_pattern in ["B", "BA"] else self.in_channels
        self.out_channels_1d = self.out_channels * (self.shuffle_scale**2) if self.shuffle_pattern in ["A", "BA"] else self.out_channels

        # Dropout Layer
        self.attention_dropout = attention_dropout
        self.dropout = nn.Dropout(p=self.attention_dropout)
        # Conv1d Layer
        self.conv1d_layer = nn.Conv1d(in_channels=self.in_channels_1d, 
                                      out_channels=self.out_channels_1d, 
                                      kernel_size=self.K, 
                                      stride=self.stride, 
                                      padding=0)

        # Flatten Layer
        self.flatten = nn.Flatten(start_dim=2)

        # Linear Projections for Q, K, V, O
        # self.num_samples_projection = self.num_samples**2 if self.sampling_type == "spatial" else self.num_samples
        # self.w_q = nn.Linear(self.num_tokens, self.num_tokens, bias=False) if self.sampling_type == "all" else nn.Linear(self.num_samples_projection, self.num_samples_projection, bias=False)
        # self.w_k = nn.Linear(self.num_tokens, self.num_tokens, bias=False) 
        # self.w_v = nn.Linear(self.num_tokens, self.num_tokens, bias=False) 
        # self.w_o = nn.Linear(self.num_tokens, self.num_tokens, bias=False)
        # Correct approach - project channels:
        self.w_q = nn.Conv1d(self.in_channels_1d, self.in_channels_1d, kernel_size=1, bias=False)
        self.w_k = nn.Conv1d(self.in_channels_1d, self.in_channels_1d, kernel_size=1, bias=False)
        self.w_v = nn.Conv1d(self.in_channels_1d, self.in_channels_1d, kernel_size=1, bias=False)
        self.w_o = nn.Conv1d(self.out_channels_1d, self.out_channels_1d, kernel_size=1, bias=False)
        
        # Pointwise Convolution Layer
        self.pointwise_conv = nn.Conv2d(in_channels=self.out_channels,
                                         out_channels=self.out_channels - 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        
    def forward(self, x): 
        # Coordinate Channels (optional) + Unshuffle + Flatten 
        x = self._add_coordinate_encoding(x) if self.coordinate_encoding else x
        x_2d = self.unshuffle_layer(x) if self.shuffle_pattern in ["B", "BA"] else x
        x = self.flatten(x_2d)

        # K, V Projections 
        k = self.dropout(self.w_k(x))
        v = self.dropout(self.w_v(x))

        if self.sampling_type == "all":    
            # Q Projection
            q = self.dropout(self.w_q(x))
            
            # ConvNN Algorithm 
            matrix_magnitude = self._calculate_distance_matrix(k, q, sqrt=True) if self.magnitude_type == 'distance' else self._calculate_similarity_matrix(k, q)
            prime = self._prime(v, matrix_magnitude, self.K, self.maximum)
             
        elif self.sampling_type == "random":
            # Select random samples
            rand_idx = torch.randperm(x.shape[2], device=x.device)[:self.num_samples]
            x_sample = x[:, :, rand_idx]

            # Q Projection
            q = self.dropout(self.w_q(x_sample))

            # ConvNN Algorithm 
            matrix_magnitude = self._calculate_distance_matrix_N(k, q, sqrt=True) if self.magnitude_type == 'distance' else self._calculate_similarity_matrix_N(k, q)
            range_idx = torch.arange(len(rand_idx), device=x.device)
            matrix_magnitude[:, rand_idx, range_idx] = self.INF_DISTANCE if self.magnitude_type == 'distance' else self.NEG_INF_DISTANCE
            prime = self._prime_N(v, matrix_magnitude, self.K, rand_idx, self.maximum)
            
        elif self.sampling_type == "spatial":
            # Get spatial sampled indices
            x_ind = torch.linspace(0 + self.sample_padding, x_2d.shape[2] - self.sample_padding - 1, self.num_samples, device=x.device).to(torch.long)
            y_ind = torch.linspace(0 + self.sample_padding, x_2d.shape[3] - self.sample_padding - 1, self.num_samples, device=x.device).to(torch.long)
            x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
            x_idx_flat, y_idx_flat = x_grid.flatten(), y_grid.flatten()
            width = x_2d.shape[2] 
            flat_indices = y_idx_flat * width + x_idx_flat  
            x_sample = x[:, :, flat_indices]

            # Q Projection
            q = self.dropout(self.w_q(x_sample))

            # ConvNN Algorithm
            matrix_magnitude = self._calculate_distance_matrix_N(k, q, sqrt=True) if self.magnitude_type == 'distance' else self._calculate_similarity_matrix_N(k, q)
            range_idx = torch.arange(len(flat_indices), device=x.device)
            matrix_magnitude[:, flat_indices, range_idx] = self.INF_DISTANCE if self.magnitude_type == 'distance' else self.NEG_INF_DISTANCE
            prime = self._prime_N(v, matrix_magnitude, self.K, flat_indices, self.maximum)
        else: 
            raise ValueError("Invalid sampling_type. Must be one of ['all', 'random', 'spatial'].")

        # Post-Processing 
        x_conv = self.conv1d_layer(prime) 
        x_drop = self.dropout(x_conv)  # Apply dropout
        x_out = self.dropout(self.w_o(x_drop))

        # Unflatten + Shuffle
        unflatten = nn.Unflatten(dim=2, unflattened_size=x_2d.shape[2:])
        x = unflatten(x_out)  # [batch_size, out_channels
        x = self.shuffle_layer(x) if self.shuffle_pattern in ["A", "BA"] else x
        x = self.pointwise_conv(x) if self.coordinate_encoding else x
        return x

    def _calculate_similarity_matrix(self, K, Q):
        k_norm = F.normalize(K, p=2, dim=1)
        q_norm = F.normalize(Q, p=2, dim=1)
        similarity_matrix = torch.bmm(k_norm.transpose(2, 1), q_norm) 
        similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0)  
        return similarity_matrix
    
    def _calculate_similarity_matrix_N(self, K, Q):
        k_norm = F.normalize(K, p=2, dim=1)
        q_norm = F.normalize(Q, p=2, dim=1)
        similarity_matrix = torch.bmm(k_norm.transpose(2, 1), q_norm)  
        similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0)
        return similarity_matrix
        
    def _calculate_distance_matrix(self, K, Q, sqrt=False):
        norm_squared_K = torch.sum(K**2, dim=1, keepdim=True) 
        norm_squared_Q = torch.sum(Q**2, dim=1, keepdim=True) 
        dot_product = torch.bmm(K.transpose(2, 1), Q)  
        dist_matrix = norm_squared_K + norm_squared_Q.transpose(2, 1) - 2 * dot_product
        dist_matrix = torch.clamp(dist_matrix, min=-1.0, max=1.0)  # remove negative values
        dist_matrix = torch.sqrt(dist_matrix) if sqrt else dist_matrix  # take square root if needed
        return dist_matrix

    def _calculate_distance_matrix_N(self, K, Q, sqrt=False):
        norm_squared_K = torch.sum(K**2, dim=1, keepdim=True).permute(0, 2, 1)
        norm_squared_Q = torch.sum(Q**2, dim=1, keepdim=True).transpose(2, 1).permute(0, 2, 1)
        dot_product = torch.bmm(K.transpose(2, 1), Q)  
        dist_matrix = norm_squared_K + norm_squared_Q - 2 * dot_product
        dist_matrix = torch.clamp(dist_matrix, min=-1.0, max=1.0)  # remove negative values
        dist_matrix = torch.sqrt(dist_matrix) if sqrt else dist_matrix  # take square root if needed
        return dist_matrix

    # def _prime_old(self, v, qk, K, maximum):
    #     b, c, t = v.shape 
    #     _, topk_indices = torch.topk(qk, k=K, dim=-1, largest = maximum)
    #     topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)
        
    #     v_expanded = v.unsqueeze(-1).expand(b, c, t, K)
    #     prime = torch.gather(v_expanded, dim=2, index=topk_indices_exp)
    #     prime = prime.reshape(b, c, -1)
    #     return prime
    
    def _prime(self, matrix, magnitude_matrix, K, maximum):
        b, c, t = matrix.shape
        topk_values, topk_indices = torch.topk(magnitude_matrix, k=K, dim=2, largest=maximum)
        topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)    
        topk_values_exp = topk_values.unsqueeze(1).expand(b, c, t, K)    
        matrix_expanded = matrix.unsqueeze(-1).expand(b, c, t, K).contiguous()
        prime = torch.gather(matrix_expanded, dim=2, index=topk_indices_exp)
        prime = topk_values_exp * prime
        prime = prime.view(b, c, -1)
        return prime
            
    # def _prime_N_old(self, v, qk, K, rand_idx, maximum):
    #     b, c, t = v.shape
    #     _, topk_indices = torch.topk(qk, k=K - 1, dim=2, largest=maximum)
    #     tk = topk_indices.shape[-1]
    #     assert K == tk + 1, "Error: K must be same as tk + 1. K == tk + 1."
    #     mapped_tensor = rand_idx[topk_indices]

    #     token_indices = torch.arange(t, device=v.device).view(1, t, 1).expand(b, t, 1)
    #     final_indices = torch.cat([token_indices, mapped_tensor], dim=2)
    #     indices_expanded = final_indices.unsqueeze(1).expand(b, c, t, K)

    #     v_expanded = v.unsqueeze(-1).expand(b, c, t, K).contiguous()
    #     prime = torch.gather(v_expanded, dim=2, index=indices_expanded) 
    #     prime = prime.reshape(b, c, -1)
    #     return prime
        
    def _prime_N(self, matrix, magnitude_matrix, K, rand_idx, maximum):
        b, c, t = matrix.shape
        
        # Get both topk_values AND topk_indices (was missing topk_values before)
        topk_values, topk_indices = torch.topk(magnitude_matrix, k=K - 1, dim=2, largest=maximum)
        tk = topk_indices.shape[-1]
        assert K == tk + 1, "Error: K must be same as tk + 1. K == tk + 1."

        # Map sample indices back to original matrix positions
        mapped_tensor = rand_idx[topk_indices]
        token_indices = torch.arange(t, device=matrix.device).view(1, t, 1).expand(b, t, 1)
        final_indices = torch.cat([token_indices, mapped_tensor], dim=2)
        indices_expanded = final_indices.unsqueeze(1).expand(b, c, t, K)

        # Create similarity weights: [1.0 for self, topk_values for neighbors]
        topk_values_exp = topk_values.unsqueeze(1).expand(b, c, t, K-1)  # Expand for K-1 neighbors
        ones = torch.ones((b, c, t, 1), device=matrix.device)            # Weight=1.0 for self
        topk_values_exp = torch.cat((ones, topk_values_exp), dim=-1)     # Combine: [self=1.0, neighbors=similarities]

        # Gather matrix values and apply similarity weighting
        matrix_expanded = matrix.unsqueeze(-1).expand(b, c, t, K).contiguous()
        prime = torch.gather(matrix_expanded, dim=2, index=indices_expanded)  
        prime = topk_values_exp * prime  # 🔥 KEY ADDITION: Apply similarity weighting
        prime = prime.view(b, c, -1)
        return prime
    
    def _add_coordinate_encoding(self, x):
        b, _, h, w = x.shape
        cache_key = f"{b}_{h}_{w}_{x.device}"

        if cache_key in self.coordinate_cache:
            expanded_grid = self.coordinate_cache[cache_key]
        else:
            y_coords_vec = torch.linspace(start=-1, end=1, steps=h, device=x.device)
            x_coords_vec = torch.linspace(start=-1, end=1, steps=w, device=x.device)

            y_grid, x_grid = torch.meshgrid(y_coords_vec, x_coords_vec, indexing='ij')
            grid = torch.stack((x_grid, y_grid), dim=0).unsqueeze(0)
            expanded_grid = grid.expand(b, -1, -1, -1)
            self.coordinate_cache[cache_key] = expanded_grid

        x_with_coords = torch.cat((x, expanded_grid), dim=1)
        return x_with_coords
    
"""(3) Attention2d"""
class Attention2d(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 num_heads,
                 shuffle_pattern,
                 shuffle_scale, 
                 coordinate_encoding=False
                 ): 
        super(Attention2d, self).__init__()
        
        ### Assertions ###
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert isinstance(num_heads, int) and num_heads > 0, "Error: num_heads must be a positive integer"
        
        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        
        # Positional Encoding (optional)
        self.coordinate_encoding = coordinate_encoding
        self.coordinate_cache = {} 
        self.in_channels = in_channels + 2 if self.coordinate_encoding else in_channels
        self.out_channels = out_channels + 2 if self.coordinate_encoding else out_channels

        # Shuffle2D/Unshuffle2D Layers
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)
        
        # Adjust Channels for PixelShuffle
        self.in_channels_1d = self.in_channels * (self.shuffle_scale**2) if self.shuffle_pattern in ["B", "BA"] else self.in_channels
        self.out_channels_1d = self.out_channels * (self.shuffle_scale**2) if self.shuffle_pattern in ["A", "BA"] else self.out_channels
                
        # 1D Attention Layer
        self.Attention1d = Attention1d(in_channels=self.in_channels_1d,
                                        out_channels=self.out_channels_1d,
                                        shuffle_pattern="NA",
                                        shuffle_scale=1,
                                        num_heads=self.num_heads
                                        )
        
        # Flatten Layer
        self.flatten = nn.Flatten(start_dim=2)
        
        # Pointwise Convolution Layer
        self.pointwise_conv = nn.Conv2d(in_channels=self.out_channels,
                                         out_channels=self.out_channels - 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        
    def forward(self, x):
        x = self._add_coordinate_encoding(x) if self.coordinate_encoding else x
        x_2d = self.unshuffle_layer(x) if self.shuffle_pattern in ["B", "BA"] else x
        x = self.flatten(x_2d) 
        x = self.Attention1d(x)
        unflatten = nn.Unflatten(dim=2, unflattened_size=x_2d.shape[2:])
        x = unflatten(x)
        x = self.shuffle_layer(x) if self.shuffle_pattern in ["A", "BA"] else x
        x = self.pointwise_conv(x) if self.coordinate_encoding else x
        return x 
    def _add_coordinate_encoding(self, x):
        b, _, h, w = x.shape
        cache_key = f"{b}_{h}_{w}_{x.device}"

        if cache_key in self.coordinate_cache:
            expanded_grid = self.coordinate_cache[cache_key]
        else:
            y_coords_vec = torch.linspace(start=-1, end=1, steps=h, device=x.device)
            x_coords_vec = torch.linspace(start=-1, end=1, steps=w, device=x.device)

            y_grid, x_grid = torch.meshgrid(y_coords_vec, x_coords_vec, indexing='ij')
            grid = torch.stack((x_grid, y_grid), dim=0).unsqueeze(0)
            expanded_grid = grid.expand(b, -1, -1, -1)
            self.coordinate_cache[cache_key] = expanded_grid

        x_with_coords = torch.cat((x, expanded_grid), dim=1)
        return x_with_coords
    
"""(4) Conv2d_ConvNN_Branching"""
class Conv2d_ConvNN_Branching(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 channel_ratio, 
                 kernel_size,
                 K, 
                 stride,
                 sampling_type, 
                 num_samples, 
                 sample_padding, 
                 shuffle_pattern, 
                 shuffle_scale, 
                 magnitude_type, 
                 coordinate_encoding=False
                 ):
        
        super(Conv2d_ConvNN_Branching, self).__init__()
        
        ### Assertions ### 
        assert K == stride, "Error: K must be same as stride. K == stride."
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert sampling_type in ["all", "random", "spatial"], "Error: sampling_type must be one of ['all', 'random', 'spatial']"
        assert int(num_samples) > 0 or int(num_samples) == -1, "Error: num_samples must be greater than 0 or -1 for all samples"
        assert (sampling_type == "all" and int(num_samples) == -1) or (sampling_type != "all" and isinstance(num_samples, int)), "Error: num_samples must be -1 for 'all' sampling or an integer for 'random' and 'spatial' sampling"
        assert sum(channel_ratio) == out_channels, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.channel_ratio = channel_ratio
        self.K = K
        self.stride = stride
        self.sampling_type = sampling_type
        self.num_samples = num_samples
        self.sample_padding = sample_padding if sampling_type == "spatial" else 0
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.magnitude_type = magnitude_type
        self.maximum = True if self.magnitude_type == 'similarity' else False

        self.coordinate_encoding = coordinate_encoding
        
        # Branch1 - Conv2d
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                nn.Conv2d(self.in_channels, 
                          self.channel_ratio[0], 
                          self.kernel_size, 
                          stride=1, 
                          padding=(self.kernel_size - 1) // 2 if self.kernel_size % 2 == 1 else self.kernel_size // 2
                          ),
                nn.ReLU()
            )
        
        # Branch2 - ConvNN
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN(self.in_channels, 
                          self.channel_ratio[1], 
                          K = self.K, 
                          stride = self.K, 
                          sampling_type = self.sampling_type,
                          num_samples = self.num_samples,
                          sample_padding = self.sample_padding,
                          shuffle_pattern=self.shuffle_pattern, 
                          shuffle_scale=self.shuffle_scale,
                          magnitude_type=self.magnitude_type, 
                          coordinate_encoding=self.coordinate_encoding), 
                nn.ReLU()
            )
        
    def forward(self, x):
        x1 = self.branch1(x) if self.channel_ratio[0] != 0 else None
        x2 = self.branch2(x) if self.channel_ratio[1] != 0 else None
        concat = torch.cat([x1, x2], dim=1) if self.channel_ratio[0] != 0 and self.channel_ratio[1] != 0 else (x1 if self.channel_ratio[0] != 0 else x2)
        return concat

"""(5) Conv2d_ConvNN_Attn_Branching"""
class Conv2d_ConvNN_Attn_Branching(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 channel_ratio, 
                 kernel_size, 
                 K,
                 stride, 
                 sampling_type, 
                 num_samples, 
                 sample_padding,
                 shuffle_pattern, 
                 shuffle_scale, 
                 magnitude_type,    
                 img_size,  
                 coordinate_encoding=False   
                ):
        super(Conv2d_ConvNN_Attn_Branching, self).__init__()

        ### Assertions ### 
        assert K == stride, "Error: K must be same as stride. K == stride."
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert sampling_type in ["all", "random", "spatial"], "Error: sampling_type must be one of ['all', 'random', 'spatial']"
        assert num_samples > 0 or num_samples == -1, "Error: num_samples must be greater than 0 or -1 for all samples"
        assert (sampling_type == "all" and num_samples == -1) or (sampling_type != "all" and isinstance(num_samples, int)), "Error: num_samples must be -1 for 'all' sampling or an integer for 'random' and 'spatial' sampling"
        assert sum(channel_ratio) == out_channels, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        
        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.channel_ratio = channel_ratio
        self.K = K
        self.stride = stride
        self.sampling_type = sampling_type
        self.num_samples = num_samples
        self.sample_padding = sample_padding if sampling_type == "spatial" else 0
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.magnitude_type = magnitude_type
        self.maximum = True if self.magnitude_type == 'similarity' else False
        self.img_size = img_size  
        
        self.coordinate_encoding = coordinate_encoding
        
        # Branch1 - Conv2d
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                nn.Conv2d(self.in_channels, 
                          self.channel_ratio[0], 
                          self.kernel_size, 
                          stride=1, 
                          padding=(self.kernel_size - 1) // 2 if self.kernel_size % 2 == 1 else self.kernel_size // 2
                         ),
                nn.ReLU()
            )
        
        # Branch2 - ConvNN_Attn
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN_Attn(self.in_channels, 
                               self.channel_ratio[1], 
                               K=self.K, 
                               stride=self.K, 
                               sampling_type=self.sampling_type,
                               num_samples=self.num_samples,
                               sample_padding=self.sample_padding,
                               shuffle_pattern=self.shuffle_pattern,
                               shuffle_scale=self.shuffle_scale,    
                               magnitude_type=self.magnitude_type,
                               img_size=self.img_size, 
                               coordinate_encoding=self.coordinate_encoding),
                nn.ReLU()
            )
        

    def forward(self, x):
        x1 = self.branch1(x) if self.channel_ratio[0] != 0 else None
        x2 = self.branch2(x) if self.channel_ratio[1] != 0 else None
        concat = torch.cat([x1, x2], dim=1) if self.channel_ratio[0] != 0 and self.channel_ratio[1] != 0 else (x1 if self.channel_ratio[0] != 0 else x2)
        return concat
    
"""(6) Attention_ConvNN_Branching"""
class Attention_ConvNN_Branching(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 channel_ratio, 
                 num_heads, 
                 K, 
                 stride, 
                 sampling_type, 
                 num_samples, 
                 sample_padding,
                 shuffle_pattern,
                 shuffle_scale,
                 magnitude_type, 
                 coordinate_encoding=False
                 ):
        super(Attention_ConvNN_Branching, self).__init__()

        ### Assertions ### 
        assert K == stride, "Error: K must be same as stride. K == stride."
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert sampling_type in ["all", "random", "spatial"], "Error: sampling_type must be one of ['all', 'random', 'spatial']"
        assert num_samples > 0 or num_samples == -1, "Error: num_samples must be greater than 0 or -1 for all samples"
        assert (sampling_type == "all" and num_samples == -1) or (sampling_type != "all" and isinstance(num_samples, int)), "Error: num_samples must be -1 for 'all' sampling or an integer for 'random' and 'spatial' sampling"
        assert sum(channel_ratio) == out_channels, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        assert isinstance(num_heads, int) and num_heads > 0, "Error: num_heads must be a positive integer"
        
        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channel_ratio = channel_ratio
        self.K = K
        self.num_heads = num_heads
        self.stride = stride
        self.sampling_type = sampling_type
        self.num_samples = num_samples
        self.sample_padding = sample_padding if sampling_type == "spatial" else 0
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.magnitude_type = magnitude_type
        self.maximum = True if self.magnitude_type == 'similarity' else False

        self.coordinate_encoding = coordinate_encoding
        
        # Branch1 - Attention2d
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                Attention2d(self.in_channels,
                            self.channel_ratio[0], 
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads, 
                            coordinate_encoding=self.coordinate_encoding),
                nn.ReLU()
            )
            
        # Branch2 - ConvNN
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN(self.in_channels, 
                          self.channel_ratio[1], 
                          K = self.K, 
                          stride = self.K, 
                          sampling_type = self.sampling_type,
                          num_samples = self.num_samples,
                          sample_padding = self.sample_padding,
                          shuffle_pattern=self.shuffle_pattern, 
                          shuffle_scale=self.shuffle_scale,
                          magnitude_type=self.magnitude_type, 
                          coordinate_encoding=self.coordinate_encoding), 
                nn.ReLU()
            )
        

    def forward(self, x):
        x1 = self.branch1(x) if self.channel_ratio[0] != 0 else None
        x2 = self.branch2(x) if self.channel_ratio[1] != 0 else None
        concat = torch.cat([x1, x2], dim=1) if self.channel_ratio[0] != 0 and self.channel_ratio[1] != 0 else (x1 if self.channel_ratio[0] != 0 else x2)
        return concat
    
"""(7) Attention_ConvNN_Attn_Branching"""
class Attention_ConvNN_Attn_Branching(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 channel_ratio, 
                 num_heads, 
                 K, 
                 stride, 
                 sampling_type, 
                 num_samples, 
                 sample_padding, 
                 shuffle_pattern, 
                 shuffle_scale, 
                 magnitude_type,
                 img_size, 
                 coordinate_encoding=False
                 ):
        
        super(Attention_ConvNN_Attn_Branching, self).__init__()
        
        ### Assertions ### 
        assert K == stride, "Error: K must be same as stride. K == stride."
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert sampling_type in ["all", "random", "spatial"], "Error: sampling_type must be one of ['all', 'random', 'spatial']"
        assert num_samples > 0 or num_samples == -1, "Error: num_samples must be greater than 0 or -1 for all samples"
        assert (sampling_type == "all" and num_samples == -1) or (sampling_type != "all" and isinstance(num_samples, int)), "Error: num_samples must be -1 for 'all' sampling or an integer for 'random' and 'spatial' sampling"
        assert sum(channel_ratio) == out_channels, "Channel ratio must add up to 2*output channels"
        assert len(channel_ratio) == 2, "Channel ratio must be of length 2"
        assert isinstance(num_heads, int) and num_heads > 0, "Error: num_heads must be a positive integer"

        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channel_ratio = channel_ratio
        self.num_heads = num_heads
        self.K = K
        self.stride = stride
        self.sampling_type = sampling_type
        self.num_samples = num_samples 
        self.sample_padding = sample_padding if sampling_type == "spatial" else 0
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.magnitude_type = magnitude_type
        self.maximum = True if self.magnitude_type == 'similarity' else False
        self.img_size = img_size  

        self.coordinate_encoding = coordinate_encoding
        
        # Branch1 - Attention2d
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                Attention2d(in_channels=self.in_channels, 
                            out_channels=self.channel_ratio[0], 
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads, 
                            coordinate_encoding=self.coordinate_encoding),
                nn.ReLU()
            )
            
        
        # Branch2 - ConvNN_Attn
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                Conv2d_NN_Attn(self.in_channels, 
                               self.channel_ratio[1], 
                               K=self.K, 
                               stride=self.K, 
                               sampling_type=self.sampling_type,
                               num_samples=self.num_samples,
                               sample_padding=self.sample_padding,
                               shuffle_pattern=self.shuffle_pattern,
                               shuffle_scale=self.shuffle_scale,    
                               magnitude_type=self.magnitude_type,
                               img_size=self.img_size, 
                               coordinate_encoding=self.coordinate_encoding),
                nn.ReLU()
            )
   
    def forward(self, x):
        x1 = self.branch1(x) if self.channel_ratio[0] != 0 else None
        x2 = self.branch2(x) if self.channel_ratio[1] != 0 else None
        concat = torch.cat([x1, x2], dim=1) if self.channel_ratio[0] != 0 and self.channel_ratio[1] != 0 else (x1 if self.channel_ratio[0] != 0 else x2)
        return concat
    
"""(8) Attention_Conv2d_Branching"""
class Attention_Conv2d_Branching(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 channel_ratio, 
                 num_heads, 
                 kernel_size,
                 shuffle_pattern, 
                 shuffle_scale,
                 coordinate_encoding=False
                 ):

        super(Attention_Conv2d_Branching, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels     
        self.channel_ratio = channel_ratio
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        
        self.coordinate_encoding = coordinate_encoding
    
        if self.channel_ratio[0] != 0:
            self.branch1 = nn.Sequential(
                Attention2d(in_channels=self.in_channels, 
                            out_channels=self.channel_ratio[0], 
                            shuffle_pattern=self.shuffle_pattern,
                            shuffle_scale=self.shuffle_scale,
                            num_heads=self.num_heads, 
                            coordinate_encoding=self.coordinate_encoding),
                nn.ReLU()
            )
            
        if self.channel_ratio[1] != 0:
            self.branch2 = nn.Sequential(
                nn.Conv2d(self.in_channels, 
                          self.channel_ratio[1], 
                          self.kernel_size, 
                          stride=1, 
                          padding=(self.kernel_size - 1) // 2 if self.kernel_size % 2 == 1 else self.kernel_size // 2
                         ),
                nn.ReLU()
            )

    def forward(self, x):
        x1 = self.branch1(x) if self.channel_ratio[0] != 0 else None
        x2 = self.branch2(x) if self.channel_ratio[1] != 0 else None
        concat = torch.cat([x1, x2], dim=1) if self.channel_ratio[0] != 0 and self.channel_ratio[1] != 0 else (x1 if self.channel_ratio[0] != 0 else x2)
        return concat
