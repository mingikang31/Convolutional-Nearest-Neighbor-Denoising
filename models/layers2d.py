"""Convolutional Nearest Neighbor Layers 2D"""

"""
Layers 2D: 
(1) Conv2d_New (Baseline Nearest Neighbor Layer w/ Pixel Shuffle and Coordinate Encoding)
(2) Conv2d_NN (Convolutional Nearest Neighbor Layer w/ Pixel Shuffle, Coordinate Encoding, Similarity and Aggregation Types, and 3 Sampling Types) 
(3) Conv2d_NN_Attn (Convolutional Nearest Neighbor Attention Layer w/ Pixel Shuffle, Coordinate Encoding, Similarity and Aggregation Types, and 3 Sampling Types) 

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
                                      padding="same")
        
    def forward(self, x): 
        x = self.unshuffle_layer(x) if self.shuffle_pattern in ["B", "BA"] else x
        print("x shape after unshuffle:", x.shape)
        x = self._add_coordinate_encoding(x) if self.aggregation_type == "Loc_Col" else x
        print("x shape after adding coordinates:", x.shape)

        # Conv2d Layer
        x = self.conv2d_layer(x)
        print("x shape after conv2d:", x.shape)
        x = self.shuffle_layer(x) if self.shuffle_pattern in ["A", "BA"] else x
        print("x shape after shuffle:", x.shape)
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
        self.num_samples = num_samples
        self.sample_padding = sample_padding if sampling_type == "spatial" else 0

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

        self.conv1d_layer = nn.Conv1d(
            in_channels = self.in_channels_1d,
            out_channels = self.out_channels_1d,
            kernel_size = self.K, 
            stride = self.stride, 
            padding = 0, 
            bias = False
        )

        # Flatten * Unflatten layers 
        self.flatten = nn.Flatten(start_dim=2)
        self.unflatten = None

        self.og_shape = None 
        self.padded_shape = None


        # Utility Variables
        self.INF = 1e5
        self.NEG_INF = -1e5

        self.lambda_param = lambda_param
        # self.lambda_param = nn.Parameter(torch.tensor(0.5), requires_grad=True)


    def forward(self, x):  
        # 1. Pixel Shuffle 
        x = self.unshuffle_layer(x) if self.shuffle_pattern in ["B", "BA"] else x
        self.og_shape = x.shape

        # 2. Add Padding 
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
            self.padded_shape = x.shape

        # 3. Add Coordinate Encoding
        x = self._add_coordinate_encoding(x) if self.coordinate_encoding else x

        # 4. Flatten 
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
            # x1 = x_sim[:, :-2, :]/math.sqrt(self.og_shape[1] - 2)
            # x2 = x_sim[:, -2:, :]/math.sqrt(2)
            # x_sim = torch.cat((x1, x2), dim=1)

            # # With Lambda 
            # x1 = self.lambda_param * x_sim[:, :-2, :]/math.sqrt(self.og_shape[1] - 2)
            # x2 = (1 - self.lambda_param) * x_sim[:, -2:, :]/math.sqrt(2)
            # x_sim = torch.cat((x1, x2), dim=1)            
            # Normalize each modality to unit variance before combining
            
            color_feats = x_sim[:, :-2, :]
            color_std = torch.std(color_feats, dim=[1,2], keepdim=True) + 1e-6
            color_norm = color_feats / color_std

            coord_feats = x_sim[:, -2:, :]  # Already in [-1,1]
            x_sim = torch.cat([self.lambda_param * color_norm, 
                            (1-self.lambda_param) * coord_feats], dim=1)

            
        
        # 4. Sampling + Similarity Calculation + Aggregation
        if self.sampling_type == "all":
            similarity_matrix = self._calculate_euclidean_matrix(x_sim) if self.magnitude_type == "euclidean" else self._calculate_cosine_matrix(x_sim)
            prime = self._prime(x, similarity_matrix, self.K, self.maximum)

        elif self.sampling_type == "random":
            rand_idx = torch.randperm(x.shape[-1], device=x.device)[:self.num_samples]
            x_sample = x_sim[:, :, rand_idx]

            similarity_matrix = self._calculate_euclidean_matrix_N(x_sim, x_sample, sqrt=True) if self.magnitude_type == "euclidean" else self._calculate_cosine_matrix_N(x_sim, x_sample)

            range_idx = torch.arange(len(rand_idx), device=x.device)
            similarity_matrix[:, rand_idx, range_idx] = self.INF if self.magnitude_type == "euclidean" else self.NEG_INF

            prime = self._prime_N(x, similarity_matrix, self.K, rand_idx, self.maximum)

        elif self.sampling_type == "spatial":
            x_ind = torch.linspace(0 + self.sample_padding, self.og_shape[-2] - self.sample_padding - 1, self.num_samples, device=x.device).to(torch.long)
            y_ind = torch.linspace(0 + self.sample_padding, self.og_shape[-1] - self.sample_padding - 1, self.num_samples, device=x.device).to(torch.long)
            x_grid, y_grid = torch.meshgrid(x_ind, y_ind, indexing='ij')
            x_idx_flat, y_idx_flat = x_grid.flatten(), y_grid.flatten()
            width = self.og_shape[-2]
            flat_indices = y_idx_flat * width + x_idx_flat
            x_sample = x_sim[:, :, flat_indices]

            similarity_matrix = self._calculate_euclidean_matrix_N(x_sim, x_sample, sqrt=True) if self.magnitude_type == "euclidean" else self._calculate_cosine_matrix_N(x_sim, x_sample)

            range_idx = torch.arange(len(flat_indices), device=x.device)    
            similarity_matrix[:, flat_indices, range_idx] = self.INF if self.magnitude_type == "euclidean" else self.NEG_INF
            prime = self._prime_N(x, similarity_matrix, self.K, flat_indices, self.maximum)
        else:
            raise NotImplementedError("Sampling Type not Implemented")
        
        # 5. Conv1d Layer
        x = self.conv1d_layer(prime)

        if not self.unflatten: 
            self.unflatten = nn.Unflatten(dim=2, unflattened_size=self.og_shape[2:])
        x = self.unflatten(x)
        x = self.shuffle_layer(x) if self.shuffle_pattern in ["A", "BA"] else x 
        return x 

    def _calculate_euclidean_matrix(self, matrix, sqrt=False):
        norm_squared = torch.sum(matrix ** 2, dim=1, keepdim=True)
        dot_product = torch.matmul(matrix.transpose(2, 1), matrix)
        dist_matrix = norm_squared + norm_squared.transpose(2, 1) - 2 * dot_product
        dist_matrix = torch.sqrt(dist_matrix) if sqrt else dist_matrix 
        dist_matrix = torch.clamp(dist_matrix, min=0.0) 

        torch.diagonal(dist_matrix, dim1=1, dim2=2).fill_(-0.1)
        return dist_matrix
    
    def _calculate_euclidean_matrix_N(self, matrix, matrix_sample, sqrt=False):
        norm_squared = torch.sum(matrix ** 2, dim=1, keepdim=True)
        norm_squared_sample = torch.sum(matrix_sample ** 2, dim=1, keepdim=True)
        dot_product = torch.bmm(matrix.transpose(1, 2), matrix_sample)
        
        dist_matrix = norm_squared.transpose(1, 2) + norm_squared_sample - 2 * dot_product
        dist_matrix = torch.clamp(dist_matrix, min=0.0) 
        dist_matrix = torch.sqrt(dist_matrix) if sqrt else dist_matrix

        return dist_matrix
    
    def _calculate_cosine_matrix(self, matrix):
        # p=2 (L2 Norm - Euclidean Distance), dim=1 (across the channels)
        norm_matrix = F.normalize(matrix, p=2, dim=1)
        similarity_matrix = torch.matmul(norm_matrix.transpose(2, 1), norm_matrix)
        similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0) 
        torch.diagonal(similarity_matrix, dim1=1, dim2=2).fill_(1.1)
        return similarity_matrix
    
    def _calculate_cosine_matrix_N(self, matrix, matrix_sample):
        # p=2 (L2 Norm - Euclidean Distance), dim=1 (across the channels)
        norm_matrix = F.normalize(matrix, p=2, dim=1) 
        norm_sample = F.normalize(matrix_sample, p=2, dim=1)
        similarity_matrix = torch.bmm(norm_matrix.transpose(2, 1), norm_sample)
        similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0) 
        return similarity_matrix
    
    def _prime(self, matrix, magnitude_matrix, K, maximum):
        b, c, t = matrix.shape

        if self.similarity_type == "Loc":
            _, topk_indices = torch.sort(magnitude_matrix, dim=2, descending=maximum, stable=True)
            topk_indices = topk_indices[:, :, :K]
        else:
            _, topk_indices = torch.topk(magnitude_matrix, k=K, dim=2, largest=maximum)

        topk_indices, _ = torch.sort(topk_indices, dim=-1)
        topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)    

        # print("topk_indices shape:", topk_indices.shape)
        # print("topk_indices: ", topk_indices)
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
        
        _, topk_indices = torch.topk(magnitude_matrix, k=K - 1, dim=2, largest=maximum)
        tk = topk_indices.shape[-1]
        assert K == tk + 1, "Error: K must be same as tk + 1. K == tk + 1."

        # print("topk_indices shape:", topk_indices.shape)
        # print("topk_indices: ", topk_indices)

        
        # Map sample indices back to original matrix positions
        mapped_tensor = rand_idx[topk_indices]
        token_indices = torch.arange(t, device=matrix.device).view(1, t, 1).expand(b, t, 1)
        final_indices = torch.cat([token_indices, mapped_tensor], dim=2)
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

    
    def _prime(self, v, qk, K, maximum):
        b, c, t = v.shape
        topk_values, topk_indices = torch.topk(qk, k=K, dim=2, largest=maximum)
        topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)
        topk_values_exp = topk_values.unsqueeze(1).expand(b, c, t, K)
        
        v_expanded = v.unsqueeze(-1).expand(b, c, t, K).contiguous()
        prime = torch.gather(v_expanded, dim=2, index=topk_indices_exp)
        prime = topk_values_exp * prime
        prime = prime.reshape(b, c, -1)
        return prime
            
    def _prime_N(self, v, qk, K, rand_idx, maximum):
        b, c, t = v.shape
        topk_values, topk_indices = torch.topk(qk, k=K-1, dim=2, largest=maximum)
        tk = topk_indices.shape[-1]
        assert K == tk + 1, "Error: K must be same as tk + 1. K == tk + 1."
        
        mapped_tensor = rand_idx[topk_indices]
        token_indices = torch.arange(t, device=v.device).view(1, t, 1).expand(b, t, 1)
        final_indices = torch.cat([token_indices, mapped_tensor], dim=2)
        indices_expanded = final_indices.unsqueeze(1).expand(b, c, t, K)

        topk_values_exp = topk_values.unsqueeze(1).expand(b, c, t, K-1)
        ones = torch.ones((b, c, t, 1), device=v.device)
        topk_values_exp = torch.cat((ones, topk_values_exp), dim=-1)

        v_expanded = v.unsqueeze(-1).expand(b, c, t, K).contiguous()
        prime = torch.gather(v_expanded, dim=2, index=indices_expanded) 
        prime = topk_values_exp * prime
        prime = prime.reshape(b, c, -1)
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
