"""Convolutional Nearest Neighbor Layers 2D"""

"""
Layers 2D: 
(1) Conv2d_New (Baseline Nearest Neighbor Layer w/ Pixel Shuffle and Coordinate Encoding)
(2) Conv2d_NN (Convolutional Nearest Neighbor Layer w/ Pixel Shuffle, Coordinate Encoding, Similarity and Aggregation Types, and 3 Sampling Types) 
(3) Conv2d_NN_Attn (Convolutional Nearest Neighbor Attention Layer w/ Pixel Shuffle, Coordinate Encoding, Similarity and Aggregation Types, and 3 Sampling Types) 
(4) Conv2d_Branching (Convolutional Nearest Neighbor Layer with Branching)

"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math

class Conv2d_New(nn.Module): 
    """Convolution 2D Nearest Neighbor Layer"""
    def __init__(self, 
                in_channels, 
                out_channels, 
                kernel_size,
                stride,
                padding, 
                shuffle_pattern, 
                shuffle_scale, 
                aggregation_type
                ): 
        
        super(Conv2d_New, self).__init__()
        
        # Assertions 
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert aggregation_type in ["Col", "Loc_Col"], "Error: aggregation_type must be one of ['Col', 'Loc_Col']"
        
        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.aggregation_type = aggregation_type 
    

        # Positional Encoding (optional)
        self.coordinate_cache = {} 

        # Shuffle2D/Unshuffle2D Layers
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)
        
        # Adjust Channels for PixelShuffle
        self.in_channels = self.in_channels * (self.shuffle_scale**2) if self.shuffle_pattern in ["B", "BA"] else self.in_channels
        self.out_channels = self.out_channels * (self.shuffle_scale**2) if self.shuffle_pattern in ["A", "BA"] else self.out_channels

        self.in_channels = self.in_channels + 2 if self.aggregation_type == "Loc_Col" else self.in_channels

    
        # Conv2d Layer
        self.conv2d_layer = nn.Conv2d(in_channels=self.in_channels, 
                                      out_channels=self.out_channels, 
                                      kernel_size=self.kernel_size, 
                                      stride=self.stride, 
                                      padding=self.padding, 
                                    #   bias=False # Only if similarity_type is "Loc" (make ConvNN exactly same as Conv2d)  
                                    )
        
    def forward(self, x): 
        # 1. Pixel Unshuffle Layer
        x = self.unshuffle_layer(x) if self.shuffle_pattern in ["B", "BA"] else x

        # 2. Add Coordinate Encoding
        x = self._add_coordinate_encoding(x) if self.aggregation_type == "Loc_Col" else x

        # 3. Conv2d Layer
        x = self.conv2d_layer(x)

        # 4. Pixel Shuffle Layer
        x = self.shuffle_layer(x) if self.shuffle_pattern in ["A", "BA"] else x
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

class Conv2d_NN(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 K, 
                 stride, 
                 padding, 
                 sampling_type, 
                 num_samples, 
                 sample_padding,
                 shuffle_pattern, 
                 shuffle_scale, 
                 magnitude_type, 
                 similarity_type, 
                 aggregation_type, 
                 lambda_param
                ):

        super(Conv2d_NN, self).__init__()

        assert K == stride, "K must be equal to stride for ConvNN"
        assert padding > 0 or padding == 0, "Cannot have Negative Padding"
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Shuffle pattern must be: Before, After, Before After, Not Applicable"
        assert magnitude_type in ["cosine", "euclidean"], "Similarity Matrix must be either cosine similarity or euclidean distance"
        assert sampling_type in ["all", "random", "spatial"], "Consider all neighbors, random neighbors, or spatial neighbors"
        assert int(num_samples) > 0 or int(num_samples) == -1, "Number of samples to consider must be greater than 0 or -1 for all samples"
        assert (sampling_type == "all" and int(num_samples) == -1) or (sampling_type != "all" and isinstance(num_samples, int)), "Number of samples must be -1 for all samples or integer for random and spatial sampling"

        assert similarity_type in ["Loc", "Col", "Loc_Col"], "Similarity Matrix based on Location, Color, or both"
        assert aggregation_type in ["Col", "Loc_Col"], "Aggregation based on Color or Location and Color"

        # Core Parameters
        self.in_channels = in_channels 
        self.out_channels = out_channels 
        self.K = K
        self.stride = stride 
        self.padding = padding 

        # 3 Sampling Types: all, random, spatial
        self.sampling_type = sampling_type
        self.num_samples = int(num_samples)
        self.sample_padding = int(sample_padding) if sampling_type == "spatial" else 0

        # Pixel Shuffling (optional) 
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale

        # Similarity Metric
        self.magnitude_type = magnitude_type
        self.maximum = True if magnitude_type == "cosine" else False

        # Similarity and Aggregation Types
        self.similarity_type = similarity_type
        self.aggregation_type = aggregation_type
        
        # Positional Encoding (optional)
        self.coordinate_encoding = True if (similarity_type in ["Loc", "Loc_Col"] or aggregation_type == "Loc_Col") else False
        self.coordinate_cache = {}

        # Pixel Shuffle Adjustments
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale) 
        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)

        self.in_channels_1d = self.in_channels * (self.shuffle_scale ** 2) if self.shuffle_pattern in ["B", "BA"] else self.in_channels
        self.out_channels_1d = self.out_channels * (self.shuffle_scale ** 2) if self.shuffle_pattern in ["A", "BA"] else self.out_channels

        self.in_channels_1d = self.in_channels_1d + 2 if self.aggregation_type == "Loc_Col" else self.in_channels_1d

        # Conv1d Layer
        self.conv1d_layer = nn.Conv1d(
            in_channels = self.in_channels_1d,
            out_channels = self.out_channels_1d,
            kernel_size = self.K, 
            stride = self.stride, 
            padding = 0, 
            # bias = False # Only if similarity_type is "Loc" (make ConvNN exactly same as Conv2d)
        )

        # Flatten * Unflatten layers 
        self.flatten = nn.Flatten(start_dim=2)
        self.unflatten = None

        # Shapes
        self.og_shape = None 
        self.padded_shape = None

        # Utility Variables
        self.INF = 1e5
        self.NEG_INF = -1e5

        self.lambda_param = lambda_param
        # self.lambda_param = nn.Parameter(torch.tensor(0.5), requires_grad=True)


    def forward(self, x):  
        # 1. Pixel Unshuffle Layer
        x = self.unshuffle_layer(x) if self.shuffle_pattern in ["B", "BA"] else x
        self.og_shape = x.shape

        # 2. Add Padding 
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
            self.padded_shape = x.shape

        # 3. Add Coordinate Encoding
        x = self._add_coordinate_encoding(x) if self.coordinate_encoding else x

        # 4. Flatten Layer
        x = self.flatten(x) 

        # 5. Similarity and Aggregation Type 
        if self.similarity_type == "Loc":
            x_sim = x[:, -2:, :]
        elif self.similarity_type == "Loc_Col":
            x_sim = x
        elif self.similarity_type == "Col" and self.aggregation_type == "Col":
            x_sim = x
        elif self.similarity_type == "Col" and self.aggregation_type == "Loc_Col":
            x_sim = x[:, :-2, :]

        if self.similarity_type in ["Loc", "Loc_Col"] and self.aggregation_type == "Col":
            x = x[:, :-2, :]
        else: 
            x = x

        if self.similarity_type == "Loc_Col":      
            # Normalize each modality to unit variance before combining
            color_feats = x_sim[:, :-2, :]
            color_std = torch.std(color_feats, dim=[1,2], keepdim=True) + 1e-6
            color_norm = color_feats / color_std

            coord_feats = x_sim[:, -2:, :]  # Already in [-1,1]
            x_sim = torch.cat([self.lambda_param * color_norm, 
                            (1-self.lambda_param) * coord_feats], dim=1)
            
        # 6. Sampling + Similarity Calculation + Aggregation
        if self.sampling_type == "all":
            similarity_matrix = self._calculate_euclidean_matrix(x_sim) if self.magnitude_type == "euclidean" else self._calculate_cosine_matrix(x_sim)
            prime = self._prime(x, similarity_matrix, self.K, self.maximum)
            
        elif self.sampling_type == "random":
            if self.num_samples > x.shape[-1]:
                x_sample = x_sim
                similarity_matrix = self._calculate_euclidean_matrix_N(x_sim, x_sample) if self.magnitude_type == "euclidean" else self._calculate_cosine_matrix_N(x_sim, x_sample)
                torch.diagonal(similarity_matrix, dim1=1, dim2=2).fill_(-0.1 if self.magnitude_type == "euclidean" else 1.1)
                prime = self._prime(x, similarity_matrix, self.K, self.maximum)

            else:
                rand_idx = torch.randperm(x.shape[-1], device=x.device)[:self.num_samples]
                x_sample = x_sim[:, :, rand_idx]
                similarity_matrix = self._calculate_euclidean_matrix_N(x_sim, x_sample) if self.magnitude_type == "euclidean" else self._calculate_cosine_matrix_N(x_sim, x_sample)
                range_idx = torch.arange(len(rand_idx), device=x.device)
                similarity_matrix[:, rand_idx, range_idx] = self.INF if self.magnitude_type == "euclidean" else self.NEG_INF
                prime = self._prime_N(x, similarity_matrix, self.K, rand_idx, self.maximum)
            

        elif self.sampling_type == "spatial":
            if self.num_samples > self.og_shape[-2]:
                x_sample = x_sim
                similarity_matrix = self._calculate_euclidean_matrix_N(x_sim, x_sample) if self.magnitude_type == "euclidean" else self._calculate_cosine_matrix_N(x_sim, x_sample)
                torch.diagonal(similarity_matrix, dim1=1, dim2=2).fill_(-0.1 if self.magnitude_type == "euclidean" else 1.1)
                prime = self._prime(x, similarity_matrix, self.K, self.maximum)
            else:
                x_ind = torch.linspace(0 + self.sample_padding, self.og_shape[-2] - self.sample_padding - 1, self.num_samples, device=x.device).to(torch.long)
                y_ind = torch.linspace(0 + self.sample_padding, self.og_shape[-1] - self.sample_padding - 1, self.num_samples, device=x.device).to(torch.long)
                x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
                x_idx_flat, y_idx_flat = x_grid.flatten(), y_grid.flatten()
                width = self.og_shape[-1]
                flat_indices = y_idx_flat * width + x_idx_flat
                x_sample = x_sim[:, :, flat_indices]

                similarity_matrix = self._calculate_euclidean_matrix_N(x_sim, x_sample) if self.magnitude_type == "euclidean" else self._calculate_cosine_matrix_N(x_sim, x_sample)

                range_idx = torch.arange(len(flat_indices), device=x.device)    
                similarity_matrix[:, flat_indices, range_idx] = self.INF if self.magnitude_type == "euclidean" else self.NEG_INF
                
                prime = self._prime_N(x, similarity_matrix, self.K, flat_indices, self.maximum)
        else:
            raise NotImplementedError("Sampling Type not Implemented")
        
        # 7. Conv1d Layer
        x = self.conv1d_layer(prime)

        # 8. Unflatten Layer
        if not self.unflatten: 
            self.unflatten = nn.Unflatten(dim=2, unflattened_size=self.og_shape[2:])
        x = self.unflatten(x)

        # 9. Pixel Shuffle Layer
        x = self.shuffle_layer(x) if self.shuffle_pattern in ["A", "BA"] else x 
        return x 

    def _calculate_euclidean_matrix(self, matrix, sqrt=False):
        norm_squared = torch.sum(matrix ** 2, dim=1, keepdim=True)
        dot_product = torch.matmul(matrix.transpose(1, 2), matrix)

        dist_matrix = norm_squared.transpose(1, 2) + norm_squared - 2 * dot_product
        dist_matrix = torch.clamp(dist_matrix, min=0.0)
        dist_matrix = torch.sqrt(dist_matrix) if sqrt else dist_matrix
        torch.diagonal(dist_matrix, dim1=1, dim2=2).fill_(-0.1)
        return dist_matrix
    
    def _calculate_euclidean_matrix_N(self, matrix, matrix_sample, sqrt=False):
        norm_squared = torch.sum(matrix ** 2, dim=1, keepdim=True)
        norm_squared_sample = torch.sum(matrix_sample ** 2, dim=1, keepdim=True)
        dot_product = torch.matmul(matrix.transpose(1, 2), matrix_sample)
        
        dist_matrix = norm_squared.transpose(1, 2) + norm_squared_sample - 2 * dot_product
        dist_matrix = torch.clamp(dist_matrix, min=0.0) 
        dist_matrix = torch.sqrt(dist_matrix) if sqrt else dist_matrix
        return dist_matrix
    
    def _calculate_cosine_matrix(self, matrix):
        norm_matrix = F.normalize(matrix, p=2, dim=1)
        similarity_matrix = torch.matmul(norm_matrix.transpose(1, 2), norm_matrix)
        similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0) 
        torch.diagonal(similarity_matrix, dim1=1, dim2=2).fill_(1.1)
        return similarity_matrix
    
    def _calculate_cosine_matrix_N(self, matrix, matrix_sample):
        norm_matrix = F.normalize(matrix, p=2, dim=1) 
        norm_sample = F.normalize(matrix_sample, p=2, dim=1)
        similarity_matrix = torch.matmul(norm_matrix.transpose(1, 2), norm_sample)
        similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0) 
        return similarity_matrix
    
    def _prime(self, matrix, magnitude_matrix, K, maximum):
        b, c, t = matrix.shape

        if self.similarity_type == "Loc":
            topk_values, topk_indices = torch.sort(magnitude_matrix, dim=2, descending=maximum, stable=True)
            topk_indices = topk_indices[:, :, :K]
            topk_indices, _ = torch.sort(topk_indices, dim=-1)
        else:
            topk_values, topk_indices = torch.topk(magnitude_matrix, k=K, dim=2, largest=maximum)

        topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)    
        matrix_expanded = matrix.unsqueeze(-1).expand(b, c, t, K).contiguous()
        prime = torch.gather(matrix_expanded, dim=2, index=topk_indices_exp)

        if self.padding > 0: 
            prime = prime.view(b, c, self.padded_shape[-2], self.padded_shape[-1], K)
            prime = prime[:, :, self.padding:-self.padding, self.padding:-self.padding, :]
            prime = prime.reshape(b, c, K * self.og_shape[-2] * self.og_shape[-1])
        else: 
            prime = prime.view(b, c, -1)

        return prime
        
    def _prime_N(self, matrix, magnitude_matrix, K, rand_idx, maximum):
        b, c, t = matrix.shape
        
        topk_values, topk_indices = torch.topk(magnitude_matrix, k=K-1, dim=2, largest=maximum)
        tk = topk_indices.shape[-1]
        assert K == tk + 1, "Error: K must be same as tk + 1. K == tk + 1."

        # Map sample indices back to original matrix positions
        mapped_tensor = rand_idx[topk_indices]
        token_indices = torch.arange(t, device=matrix.device).view(1, t, 1).expand(b, t, 1)
        final_indices = torch.cat([token_indices, mapped_tensor], dim=2)
        if self.similarity_type == "Loc":
            final_indices, _ = torch.sort(final_indices, dim=-1)
        indices_expanded = final_indices.unsqueeze(1).expand(b, c, t, K)

        # Gather matrix values and apply similarity weighting
        matrix_expanded = matrix.unsqueeze(-1).expand(b, c, t, K).contiguous()
        prime = torch.gather(matrix_expanded, dim=2, index=indices_expanded)  

        if self.padding > 0:
            prime = prime.view(b, c, self.padded_shape[-2], self.padded_shape[-1], K)
            prime = prime[:, :, self.padding:-self.padding, self.padding:-self.padding, :]
            prime = prime.reshape(b, c, K * self.og_shape[-2] * self.og_shape[-1])
        else:
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
        return x_with_coords ### Last two channels are coordinate channels 

"""(2) Conv2d_NN_Attn (All, Random, Spatial Sampling)""" ## TODO Need to work on this layer 
class Conv2d_NN_Attn(nn.Module): 
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 K, 
                 stride, 
                 padding, 
                 sampling_type, 
                 num_samples, 
                 sample_padding,
                 shuffle_pattern, 
                 shuffle_scale, 
                 magnitude_type, 
                 aggregation_type, 
                 attention_dropout
                ): 

        super(Conv2d_NN_Attn, self).__init__()
        
        # Assertions 
        assert K == stride, "Error: K must be same as stride. K == stride."
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["cosine", "euclidean"], "Similarity Matrix must be either cosine similarity or euclidean distance"
        assert sampling_type in ["all", "random", "spatial"], "Error: sampling_type must be one of ['all', 'random', 'spatial']"
        assert int(num_samples) > 0 or int(num_samples) == -1, "Error: num_samples must be greater than 0 or -1 for all samples"
        assert (sampling_type == "all" and int(num_samples) == -1) or (sampling_type != "all" and isinstance(num_samples, int)), "Error: num_samples must be -1 for 'all' sampling or an integer for 'random' and 'spatial' sampling"

        assert aggregation_type in ["Col", "Loc_Col"], "Error: aggregation_type must be one of ['Col', 'Loc_Col']"

        # Core Parameters 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.stride = stride
        self.padding = padding

        # 3 Sampling Types: all, random, spatial
        self.sampling_type = sampling_type
        self.num_samples = int(num_samples)
        self.sample_padding = int(sample_padding) if sampling_type == "spatial" else 0

        # Pixel Shuffling (optional) 
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale

        # Similarity Metric 
        self.magnitude_type = magnitude_type
        self.maximum = True if self.magnitude_type == 'cosine' else False

        # Positional Encoding 
        self.coordinate_encoding = True if aggregation_type == "Loc_Col" else False
        self.coordinate_cache = {}

        # Pixel Shuffle Adjustments
        self.shuffle_layer = nn.PixelShuffle(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = nn.PixelUnshuffle(downscale_factor=self.shuffle_scale)

        self.in_channels_1d = self.in_channels * (self.shuffle_scale ** 2) if self.shuffle_pattern in ["B", "BA"] else self.in_channels
        self.out_channels_1d = self.out_channels * (self.shuffle_scale ** 2) if self.shuffle_pattern in ["A", "BA"] else self.out_channels

        self.in_channels_1d = self.in_channels_1d + 2 if self.coordinate_encoding else self.in_channels_1d

        # Conv1d Layer 
        self.conv1d_layer = nn.Conv1d(
            in_channels = self.in_channels_1d,
            out_channels = self.out_channels_1d,
            kernel_size = self.K, 
            stride = self.stride, 
            padding = 0, 
        )

        # Flatten * Unflatten layers
        self.flatten = nn.Flatten(start_dim=2)
        self.unflatten = None

        # Shapes 
        self.og_shape = None
        self.padded_shape = None

        # Utility Variables
        self.INF = 1e5
        self.NEG_INF = -1e5

        # Attention Parameters 
        self.attention_dropout = attention_dropout
        self.dropout = nn.Dropout(p=self.attention_dropout)

        # Query, Key, Value, Output Projections
        self.w_q = nn.Conv1d(self.in_channels_1d, self.in_channels_1d, kernel_size=1, stride=1, bias=False)
        self.w_k = nn.Conv1d(self.in_channels_1d, self.in_channels_1d, kernel_size=1, stride=1, bias=False)
        # self.w_v = nn.Conv1d(self.in_channels_1d, self.in_channels_1d, kernel_size=1, stride=1, bias=False)

        ## TODO Try with sequential layer for projections for ConvNN Attention 
        self.w_v = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels_1d, out_channels=8, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=self.in_channels_1d, kernel_size=1, stride=1, bias=False),
        )
        
        self.w_o = nn.Conv1d(self.out_channels_1d, self.out_channels_1d, kernel_size=1, stride=1, bias=False)

    def forward(self, x): 
        # 1. Pixel Unshuffle Layer 
        x = self.unshuffle_layer(x) if self.shuffle_pattern in ["B", "BA"] else x
        self.og_shape = x.shape

        # 2. Add Padding
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
            self.padded_shape = x.shape

        # 3. Add Coordinate Encoding
        x = self._add_coordinate_encoding(x) if self.coordinate_encoding else x

        # 4. Flatten Layer
        x = self.flatten(x) 

        # 5. K, V Projections
        # k = self.w_k(x)
        k = x
        v = self.w_v(x)

        if self.sampling_type == "all":
            # Q Projection
            # q = self.w_q(x)
            q = x

            similarity_matrix = self._calculate_euclidean_matrix(k, q) if self.magnitude_type == 'euclidean' else self._calculate_cosine_matrix(k, q)
            prime = self._prime(v, similarity_matrix, self.K, self.maximum)
            
        elif self.sampling_type == "random":
            if self.num_samples > x.shape[-1]:
                x_sample = x
                # q = self.w_q(x_sample)
                q = x_sample
                similarity_matrix = self._calculate_euclidean_matrix_N(k, q) if self.magnitude_type == 'euclidean' else self._calculate_cosine_matrix_N(k, q)
                torch.diagonal(similarity_matrix, dim1=1, dim2=2).fill_(-0.1 if self.magnitude_type == 'euclidean' else 1.1)
                prime = self._prime(v, similarity_matrix, self.K, self.maximum)

            else: 
                rand_idx = torch.randperm(x.shape[-1], device=x.device)[:self.num_samples]
                x_sample = x[:, :, rand_idx]

                # Q Projection
                # q = self.w_q(x_sample)
                q = x_sample

                similarity_matrix = self._calculate_euclidean_matrix_N(k, q) if self.magnitude_type == 'euclidean' else self._calculate_cosine_matrix_N(k, q)
                
                range_idx = torch.arange(len(rand_idx), device=x.device)
                similarity_matrix[:, rand_idx, range_idx] = self.INF if self.magnitude_type == 'euclidean' else self.NEG_INF
                
                prime = self._prime_N(v, similarity_matrix, self.K, rand_idx, self.maximum)
            
        elif self.sampling_type == "spatial":
            if self.num_samples > self.og_shape[-2]:
                x_sample = x
                # q = self.w_q(x_sample)
                q = x
                similarity_matrix = self._calculate_euclidean_matrix_N(k, q) if self.magnitude_type == "euclidean" else self._calculate_cosine_matrix_N(k, q)
                torch.diagonal(similarity_matrix, dim1=1, dim2=2).fill_(-0.1 if self.magnitude_type == 'euclidean' else 1.1)
                prime = self._prime(v, similarity_matrix, self.K, self.maximum)
            else:
                    
                x_ind = torch.linspace(0 + self.sample_padding, self.og_shape[-2] - self.sample_padding - 1, self.num_samples, device=x.device).to(torch.long)
                y_ind = torch.linspace(0 + self.sample_padding, self.og_shape[-1] - self.sample_padding - 1, self.num_samples, device=x.device).to(torch.long)
                x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
                x_idx_flat, y_idx_flat = x_grid.flatten(), y_grid.flatten()
                width = self.og_shape[-2]
                flat_indices = y_idx_flat * width + x_idx_flat
                x_sample = x[:, :, flat_indices]

                # Q Projection
                # q = self.w_q(x_sample)
                q = x_sample

                similarity_matrix = self._calculate_euclidean_matrix_N(k, q) if self.magnitude_type == "euclidean" else self._calculate_cosine_matrix_N(k, q)

                range_idx = torch.arange(len(flat_indices), device=x.device)    
                similarity_matrix[:, flat_indices, range_idx] = self.INF if self.magnitude_type == "euclidean" else self.NEG_INF
                
                prime = self._prime_N(x, similarity_matrix, self.K, flat_indices, self.maximum)
        else:
            raise NotImplementedError("Sampling Type not Implemented")

        # 7. Conv1d Layer
        x = self.conv1d_layer(prime)

        # 8. Output Projection
        x = self.w_o(x)

        # 9. Unflatten Layer
        if not self.unflatten: 
            self.unflatten = nn.Unflatten(dim=2, unflattened_size=self.og_shape[2:])
        x = self.unflatten(x)

        # 10. Pixel Shuffle Layer
        x = self.shuffle_layer(x) if self.shuffle_pattern in ["A", "BA"] else x
        return x

    
    def _calculate_euclidean_matrix(self, K, Q, sqrt=False):
        k_norm_squared = torch.sum(K**2, dim=1, keepdim=True) 
        q_norm_squared = torch.sum(Q**2, dim=1, keepdim=True) 

        dot_product = torch.matmul(K.transpose(1, 2), Q)
        
        dist_matrix = k_norm_squared.transpose(1, 2) + q_norm_squared - 2 * dot_product
        dist_matrix = torch.clamp(dist_matrix, min=0.0)
        dist_matrix = torch.sqrt(dist_matrix) if sqrt else dist_matrix 
        torch.diagonal(dist_matrix, dim1=1, dim2=2).fill_(-0.1)
        
        return dist_matrix
    
    def _calculate_euclidean_matrix_N(self, K, Q, sqrt=False):
        k_norm_squared = torch.sum(K ** 2, dim=1, keepdim=True)
        q_norm_squared = torch.sum(Q ** 2, dim=1, keepdim=True)
        
        dot_product = torch.matmul(K.transpose(1, 2), Q)

        dist_matrix = k_norm_squared.transpose(1, 2) + q_norm_squared - 2 * dot_product
        dist_matrix = torch.clamp(dist_matrix, min=0.0) 
        dist_matrix = torch.sqrt(dist_matrix) if sqrt else dist_matrix

        return dist_matrix
    
    def _calculate_cosine_matrix(self, K, Q):
        k_norm = F.normalize(K, p=2, dim=1)
        q_norm = F.normalize(Q, p=2, dim=1)
        similarity_matrix = torch.matmul(k_norm.transpose(1, 2), q_norm)
        similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0)
        torch.diagonal(similarity_matrix, dim1=1, dim2=2).fill_(1.1)
        return similarity_matrix 
    
    def _calculate_cosine_matrix_N(self, K, Q):
        norm_k = F.normalize(K, p=2, dim=1)
        norm_q = F.normalize(Q, p=2, dim=1)
        similarity_matrix = torch.bmm(norm_k.transpose(1, 2), norm_q)
        similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0)
        return similarity_matrix
    
    def _prime(self, v, qk, K, maximum):
        b, c, t = v.shape
        topk_values, topk_indices = torch.topk(qk, k=K, dim=2, largest=maximum)
        topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)    
        topk_values_exp = topk_values.unsqueeze(1).expand(b, c, t, K)

        v_expanded = v.unsqueeze(-1).expand(b, c, t, K).contiguous()
        prime = torch.gather(v_expanded, dim=2, index=topk_indices_exp)
        prime = prime * topk_values_exp

        if self.padding > 0: 
            prime = prime.view(b, c, self.padded_shape[-2], self.padded_shape[-1], K)
            prime = prime[:, :, self.padding:-self.padding, self.padding:-self.padding, :]
            prime = prime.reshape(b, c, K * self.og_shape[-2] * self.og_shape[-1])
        else: 
            prime = prime.view(b, c, -1)

        return prime
        
    def _prime_N(self, v, qk, K, rand_idx, maximum):
        b, c, t = v.shape

        topk_values, topk_indices = torch.topk(qk, k=K - 1, dim=2, largest=maximum)
        tk = topk_indices.shape[-1]
        assert K == tk + 1, "Error: K must be same as tk + 1. K == tk + 1."
        
        # Map sample indices back to original matrix positions
        mapped_tensor = rand_idx[topk_indices]
        token_indices = torch.arange(t, device=v.device).view(1, t, 1).expand(b, t, 1)
        final_indices = torch.cat([token_indices, mapped_tensor], dim=2)
        indices_expanded = final_indices.unsqueeze(1).expand(b, c, t, K)

        topk_values_exp = topk_values.unsqueeze(1).expand(b, c, t, K - 1)
        ones = torch.ones((b, c, t, 1), device=v.device)
        topk_values_exp = torch.cat([ones, topk_values_exp], dim=-1)

        # Gather matrix values and apply similarity weighting
        v_expanded = v.unsqueeze(-1).expand(b, c, t, K).contiguous()
        prime = torch.gather(v_expanded, dim=2, index=indices_expanded)
        prime = prime * topk_values_exp

        if self.padding > 0:
            prime = prime.view(b, c, self.padded_shape[-2], self.padded_shape[-1], K)
            prime = prime[:, :, self.padding:-self.padding, self.padding:-self.padding, :]
            prime = prime.reshape(b, c, K * self.og_shape[-2] * self.og_shape[-1])
        else:
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


class Conv2d_Branching(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 K, 
                 stride, 
                 padding, 
                 sampling_type, 
                 num_samples, 
                 sample_padding, 
                 shuffle_pattern, 
                 shuffle_scale, 
                 magnitude_type, 
                 similarity_type, 
                 aggregation_type, 
                 lambda_param, 
                 branch_ratio=0.5
                 ):
        super(Conv2d_Branching, self).__init__()

        assert 0 <= branch_ratio <= 1, "Branch ratio must be between 0 and 1"

        self.branch_ratio = branch_ratio
        self.in_channels = in_channels
        self.out_channels_1 = int(out_channels * branch_ratio)
        self.out_channels_2 = out_channels - self.out_channels_1

        if self.branch_ratio != 0:
            self.branch1 = Conv2d_NN(
                in_channels=self.in_channels,
                out_channels=self.out_channels_1,
                K=K,
                stride=K,
                padding=padding,
                sampling_type=sampling_type,
                num_samples=num_samples,
                sample_padding=sample_padding,
                shuffle_pattern=shuffle_pattern,
                shuffle_scale=shuffle_scale,
                magnitude_type=magnitude_type,
                similarity_type=similarity_type,
                aggregation_type=aggregation_type,
                lambda_param=lambda_param
            )
        if self.branch_ratio != 1:
            self.branch2 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels_2,
                kernel_size=kernel_size,
                stride=1,
                padding="same"
            )

            
        self.pointwise_conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # self.channel_shuffle = nn.ChannelShuffle(groups=2) # Optional Channel Shuffle - not in use

    def forward(self, x):
        if self.branch_ratio == 0:
            x = self.branch2(x)
            out = self.pointwise_conv(x)
            return out
        if self.branch_ratio == 1:
            x = self.branch1(x)
            out = self.pointwise_conv(x)
            return out
        
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        # x1 = self.branch(x[:, :self.in_channels_1, :, :]
        # x2 = self.conv2d(x[:, self.in_channels_2:, :, :])
        out = torch.cat((x1, x2), dim=1)
        out = self.pointwise_conv(out)
        # print("Out Shape:", out.shape)
        return out


class Conv2d_Attn_Branching(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 K, 
                 stride, 
                 padding, 
                 sampling_type, 
                 num_samples, 
                 sample_padding, 
                 shuffle_pattern, 
                 shuffle_scale, 
                 magnitude_type, 
                 aggregation_type, 
                 attention_dropout,
                 branch_ratio=0.5
                 ):
        super(Conv2d_Attn_Branching, self).__init__()

        assert 0 <= branch_ratio <= 1, "Branch ratio must be between 0 and 1"

        self.branch_ratio = branch_ratio
        self.in_channels = in_channels
        # self.in_channels_1 = int(in_channels * branch_ratio)
        # self.in_channels_2 = in_channels - self.in_channels_1
        self.out_channels_1 = int(out_channels * branch_ratio)
        self.out_channels_2 = out_channels - self.out_channels_1

        if self.branch_ratio != 0:
            self.branch1 = Conv2d_NN_Attn(
                in_channels=self.in_channels,
                out_channels=self.out_channels_1,
                K=K,
                stride=K,
                padding=padding,
                sampling_type=sampling_type,
                num_samples=num_samples,
                sample_padding=sample_padding,
                shuffle_pattern=shuffle_pattern,
                shuffle_scale=shuffle_scale,
                magnitude_type=magnitude_type,
                aggregation_type=aggregation_type,
                attention_dropout=attention_dropout
            )
        if self.branch_ratio != 1:
            self.branch2 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels_2,
                kernel_size=kernel_size,
                stride=1,
                padding="same"
            )

            
        self.pointwise_conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # self.channel_shuffle = nn.ChannelShuffle(groups=2) # Optional Channel Shuffle - not in use

    def forward(self, x):
        if self.branch_ratio == 0:
            x = self.branch2(x)
            out = self.pointwise_conv(x)
            return out
        if self.branch_ratio == 1:
            x = self.branch1(x)
            out = self.pointwise_conv(x)
            return out
        
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        # x1 = self.branch(x[:, :self.in_channels_1, :, :]
        # x2 = self.conv2d(x[:, self.in_channels_2:, :, :])
        out = torch.cat((x1, x2), dim=1)
        out = self.pointwise_conv(out)
        # print("Out Shape:", out.shape)
        return out


