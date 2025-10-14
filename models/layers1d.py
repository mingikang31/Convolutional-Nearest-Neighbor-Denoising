"""Convolutional Nearest Neighbor Layers 1D"""

"""
Layers 1D: 

(*) PixelShuffle1D
(*) PixelUnshuffle1D

(1) Conv1d_New (Baseline Convolutional Layer with PixelShuffle1D and Coordinate Encoding options)
(2) Conv1d_NN (All, Random, Spatial Sampling) 
(3) Conv1d_NN_Attn (All, Random, Spatial Sampling) 
(4) Conv1d_Branching (Conv1d_NN + Conv1d Hybrid Layer)
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 


"""(*) PixelShuffle1D"""
class PixelShuffle1D(nn.Module): 
    """
    1D Pixel Shuffle Layer for Convolutional Neural Networks.
    
    Attributes: 
        upscale_factor (int): Upscale factor for pixel shuffle. 
        
    Notes:
        Input's channel size must be divisible by the upscale factor. 
    """
    
    def __init__(self, upscale_factor):
        """ 
        Initializes the PixelShuffle1D module.
        
        Parameters:
            upscale_factor (int): Upscale factor for pixel shuffle.
        """
        super(PixelShuffle1D, self).__init__()
        
        self.upscale_factor = upscale_factor

    def forward(self, x): 
        batch_size, channel_len, token_len = x.shape[0], x.shape[1], x.shape[2]
        
        output_channel_len = channel_len / self.upscale_factor 
        if output_channel_len.is_integer() == False: 
            raise ValueError('Input channel length must be divisible by upscale factor')
        output_channel_len = int(output_channel_len)
        
        output_token_len = int(token_len * self.upscale_factor)
        
        x = torch.reshape(x, (batch_size, output_channel_len, output_token_len)).contiguous()
        
        return x 

"""(*) PixelUnshuffle1D"""
class PixelUnshuffle1D(nn.Module):  
    """
    1D Pixel Unshuffle Layer for Convolutional Neural Networks.
    
    Attributes:
        downscale_factor (int): Downscale factor for pixel unshuffle.
        
    Note:
        Input's token size must be divisible by the downscale factor
    
    """
    
    def __init__(self, downscale_factor):
        """
        Intializes the PixelUnshuffle1D module.
        
        Parameters:
            downscale_factor (int): Downscale factor for pixel unshuffle.
        """
        super(PixelUnshuffle1D, self).__init__()
        
        self.downscale_factor = downscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        channel_len = x.shape[1]
        token_len = x.shape[2]

        output_channel_len = int(channel_len * self.downscale_factor)
        output_token_len = token_len / self.downscale_factor
        
        if output_token_len.is_integer() == False:
            raise ValueError('Input token length must be divisible by downscale factor')
        output_token_len = int(output_token_len)
        
        x = torch.reshape(x, (batch_size, output_channel_len, output_token_len)).contiguous()
        
        return x 


class Conv1d_New(nn.Module): 
    """Convolutional Nearest Neighbor Layer 1D"""
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
        super(Conv1d_New, self).__init__()

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
        self.coordinate_cache = {}  # Cache for coordinate encoding

        # Shuffle1D/Unshuffle1D Layer
        self.shuffle_layer = PixelShuffle1D(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = PixelUnshuffle1D(downscale_factor=self.shuffle_scale)

        # Adjust Channels for PixelShuffle
        self.in_channels = self.in_channels * shuffle_scale if self.shuffle_pattern in ["BA", "B"] else self.in_channels
        self.out_channels = self.out_channels * shuffle_scale if self.shuffle_pattern in ["BA", "A"] else self.out_channels

        self.in_channels = self.in_channels + 1 if self.aggregation_type == "Loc_Col" else self.in_channels

        # Conv1d Layer
        self.conv1d_layer = nn.Conv1d(in_channels=self.in_channels, 
                                      out_channels=self.out_channels, 
                                      kernel_size=self.kernel_size, 
                                      stride=self.stride, 
                                      padding=self.padding, 
                                      #   bias=False # Only if similarity_type is "Loc" (make ConvNN exactly same as Conv1d)  
                                      )

    def forward(self, x):
        # 1. Pixel Unshuffle Layer
        x = self.unshuffle_layer(x) if self.shuffle_pattern in ["B", "BA"] else x

        # 2. Add Coordinate Encoding
        x = self._add_coordinate_encoding(x) if self.aggregation_type == "Loc_Col" else x

        # 3. Conv1d Layer
        x = self.conv1d_layer(x)

        # 4. Pixel Shuffle Layer
        x = self.shuffle_layer(x) if self.shuffle_pattern in ["A", "BA"] else x
        return x
    
    def _add_coordinate_encoding(self, x):
        b, c, t = x.shape 
        cache_key = f"{b}_{t}_{x.device}"
        if cache_key in self.coordinate_cache: 
            expanded_coords = self.coordinate_cache[cache_key]
        else: 
            coords_vec = torch.linspace(start=-1, end=1, steps=t, device=x.device).unsqueeze(0).expand(b, -1) 
            expanded_coords = coords_vec.unsqueeze(1).expand(b, -1, -1) 
            self.coordinate_cache[cache_key] = expanded_coords

        x_with_coords = torch.cat([x, expanded_coords], dim=1) 
        return x_with_coords
    
"""(1) Conv1d_NN (All, Random, Spatial Sampling)"""
class Conv1d_NN(nn.Module): 
    """Convolutional Nearest Neighbor Layer 1D"""
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
        
        super(Conv1d_NN, self).__init__()
        
        # Assertions 
        assert K == stride, "Error: K must be same as stride. K == stride."
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["cosine", "euclidean"], "Error: magnitude_type must be one of ['cosine', 'euclidean']"
        assert sampling_type in ["all", "random", "spatial"], "Error: sampling_type must be one of ['all', 'random', 'spatial']"
        assert int(num_samples) > 0 or int(num_samples) == -1, "Error: num_samples must be greater than 0 or -1 for all samples"
        assert (sampling_type == "all" and num_samples == -1) or (sampling_type != "all" and isinstance(num_samples, int)), "Error: num_samples must be -1 for 'all' sampling or an integer for 'random' and 'spatial' sampling"
        assert similarity_type in ["Loc", "Col", "Loc_Col"], "Error: similarity_type must be one of ['Loc', 'Col', 'Loc_Col']"
        assert aggregation_type in ["Loc", "Col", "Loc_Col"], "Error: aggregation_type must be one of ['Loc', 'Col', 'Loc_Col']"

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

        # Similarity and Aggregation Types 
        self.similarity_type = similarity_type
        self.aggregation_type = aggregation_type

        # Positional Encoding (optional)
        self.coordinate_encoding = True if self.similarity_type in ["Loc", "Loc_Col"] or self.aggregation_type in ["Loc_Col"] else False
        self.coordinate_cache = {} 

        # Pixel Shuffle Adjustments 
        self.in_channels = self.in_channels * shuffle_scale if self.shuffle_pattern in ["BA", "B"] else self.in_channels
        self.out_channels = self.out_channels * shuffle_scale if self.shuffle_pattern in ["BA", "A"] else self.out_channels

        self.in_channels = self.in_channels + 1 if self.aggregation_type == "Loc_Col" else self.in_channels

        # Conv1d Layer
        self.conv1d_layer = nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.K, 
            stride=self.stride, 
            padding=0, 
             # bias=False # Only if similarity_type is "Loc" (make ConvNN exactly same as Conv2d)
            )

        # Shuffle1D/Unshuffle1D Layer
        self.shuffle_layer = PixelShuffle1D(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = PixelUnshuffle1D(downscale_factor=self.shuffle_scale)

        # Shapes 
        self.og_shape = None 
        self.padded_shape = None

        # Utility Variables 
        self.INF = 1e5
        self.NEG_INF = -1e5

        self.lambda_param = lambda_param
        # self.lambda_param = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x):
        # 1. Pixel Unshuffle 
        x = self.unshuffle_layer(x) if self.shuffle_pattern in ["B", "BA"] else x
        self.og_shape = x.shape

        # 2. Add Padding 
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding), mode='constant', value=0)
            self.padded_shape = x.shape

        # 3. Add Coordinate Encoding 
        x = self._add_coordinate_encoding(x) if self.coordinate_encoding else x

        # 4. Similarity and Aggregation Type 
        if self.similarity_type == "Loc":
            x_sim = x[:, -1:, :]
        elif self.similarity_type == "Loc_Col":
            x_sim = x
        elif self.similarity_type == "Col" and self.aggregation_type == "Col":
            x_sim = x
        elif self.similarity_type == "Col" and self.aggregation_type == "Loc_Col":
            x_sim = x[:, :-1, :]

        if self.similarity_type in ["Loc", "Loc_Col"] and self.aggregation_type == "Col":
            x = x[:, :-1, :]
        else: 
            x = x 

        if self.similarity_type == "Loc_Col":
            color_feats = x_sim[:, :-1, :]
            color_std = torch.std(color_feats, dim=[1, 2], keepdim=True) + 1e-6
            color_norm = color_feats / color_std

            coord_feats = x_sim[:, -1:, :]
            x_sim = torch.cat([self.lambda_param * color_norm, 
                               (1 - self.lambda_param) * coord_feats], dim=1)

        # 5. Sampling + Similarity Calculation + Aggregation 
        if self.sampling_type == "all":
            similarity_matrix = self._calculate_euclidean_matrix(x_sim) if self.magnitude_type == 'euclidean' else self._calculate_cosine_matrix(x_sim)
            prime = self._prime(x, similarity_matrix, self.K, self.maximum) 

        elif self.sampling_type == "random":
            rand_idx = torch.randperm(x.shape[-1], device=x.device)[:self.num_samples]
            x_sample = x_sim[:, :, rand_idx]

            similarity_matrix = self._calculate_euclidean_matrix_N(x_sim, x_sample) if self.magnitude_type == 'euclidean' else self._calculate_cosine_matrix_N(x_sim, x_sample)

            range_idx = torch.arange(len(rand_idx), device=x.device)
            similarity_matrix[:, rand_idx, range_idx] = self.INF if self.magnitude_type == 'euclidean' else self.NEG_INF

            prime = self._prime_N(x, similarity_matrix, self.K, rand_idx, self.maximum)

        elif self.sampling_type == "spatial":
            spat_idx = torch.linspace(0 + self.sample_padding, x.shape[-1] - self.sample_padding - 1, self.num_samples, device=x.device).to(torch.long)
            x_sample = x_sim[:, :, spat_idx]

            similarity_matrix = self._calculate_euclidean_matrix_N(x_sim, x_sample) if self.magnitude_type == 'euclidean' else self._calculate_cosine_matrix_N(x_sim, x_sample)

            range_idx = torch.arange(len(spat_idx), device=x.device)
            similarity_matrix[:, spat_idx, range_idx] = self.INF if self.magnitude_type == 'euclidean' else self.NEG_INF

            prime = self._prime_N(x, similarity_matrix, self.K, spat_idx, self.maximum)
        else: 
            raise ValueError("Invalid sampling_type. Must be one of ['all', 'random', 'spatial'].")

        # 6. Conv1d Layer
        x = self.conv1d_layer(prime)

        # 7. Pixel Shuffle
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
            prime = prime.view(b, c, self.padded_shape[-1], K)
            prime = prime[:, :, self.padding:-self.padding, :]
            prime = prime.reshape(b, c, K * self.og_shape[-1])
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
            prime = prime.view(b, c, self.padded_shape[-1], K)
            prime = prime[:, :, self.padding:-self.padding, :]
            prime = prime.reshape(b, c, K * self.og_shape[-1])
        else:
            prime = prime.view(b, c, -1)
        return prime

    def _add_coordinate_encoding(self, x):
        b, c, t = x.shape 
        cache_key = f"{b}_{t}_{x.device}"
        if cache_key in self.coordinate_cache: 
            expanded_coords = self.coordinate_cache[cache_key]
        else: 
            coords_vec = torch.linspace(start=-1, end=1, steps=t, device=x.device).unsqueeze(0).expand(b, -1) 
            expanded_coords = coords_vec.unsqueeze(1).expand(b, -1, -1) 
            self.coordinate_cache[cache_key] = expanded_coords

        x_with_coords = torch.cat([x, expanded_coords], dim=1) 
        return x_with_coords

"""(2) Conv1d_NN_Attn (All, Random, Spatial Sampling)"""
class Conv1d_NN_Attn(nn.Module): 
    """Convolutional Nearest Neighbor Layer 1D"""
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
                 coordinate_encoding, 
                 num_tokens, 
                 attention_dropout
                 ):
        
        super(Conv1d_NN_Attn, self).__init__()
        
        # Assertions 
        assert K == stride, "Error: K must be same as stride. K == stride."
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["cosine", "euclidean"], "Error: magnitude_type must be one of ['cosine', 'euclidean']"
        assert sampling_type in ["all", "random", "spatial"], "Error: sampling_type must be one of ['all', 'random', 'spatial']"
        assert int(num_samples) > 0 or int(num_samples) == -1, "Error: num_samples must be greater than 0 or -1 for all samples"
        assert (sampling_type == "all" and num_samples == -1) or (sampling_type != "all" and isinstance(num_samples, int)), "Error: num_samples must be -1 for 'all' sampling or an integer for 'random' and 'spatial' sampling"

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

        # Positional Encoding (optional)
        self.coordinate_encoding = coordinate_encoding
        self.coordinate_cache = {} 

        # Pixel Shuffle Adjustments 
        self.shuffle_layer = PixelShuffle1D(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = PixelUnshuffle1D(downscale_factor=self.shuffle_scale)

        self.in_channels = self.in_channels * shuffle_scale if self.shuffle_pattern in ["BA", "B"] else self.in_channels
        self.out_channels = self.out_channels * shuffle_scale if self.shuffle_pattern in ["BA", "A"] else self.out_channels

        self.in_channels = self.in_channels + 1 if self.coordinate_encoding else self.in_channels

        # Conv1d Layer
        self.conv1d_layer = nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.K, 
            stride=self.stride, 
            padding=0, 
            #   bias=False # Make ConvNN exactly same as Conv1d
            )

        # Shapes 
        self.og_shape = None 
        self.padded_shape = None

        # Utility Variables 
        self.INF = 1e5
        self.NEG_INF = -1e5

        # Attention Parameters 
        self.num_tokens = num_tokens // self.shuffle_scale if self.shuffle_pattern in ["BA", "A"] else num_tokens
        self.num_tokens_padded = self.num_tokens + padding*2 if self.padding > 0 else self.num_tokens
        self.attention_dropout = attention_dropout
        self.dropout = nn.Dropout(p=self.attention_dropout)

        # Attention Linear Projections
        self.w_q = nn.Linear(self.num_tokens_padded, self.num_tokens_padded, bias=False) if self.sampling_type == "all" else nn.Linear(self.num_samples, self.num_samples, bias=False)
        self.w_k = nn.Linear(self.num_tokens_padded, self.num_tokens_padded, bias=False)
        self.w_v = nn.Linear(self.num_tokens_padded, self.num_tokens_padded, bias=False)
        self.w_o = nn.Linear(self.num_tokens, self.num_tokens, bias=False)

    def forward(self, x):
        # 1. Pixel Unshuffle 
        x = self.unshuffle_layer(x) if self.shuffle_pattern in ["B", "BA"] else x
        self.og_shape = x.shape

        # 2. Add Padding 
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding), mode='constant', value=0)
            self.padded_shape = x.shape

        # 3. Add Coordinate Encoding 
        x = self._add_coordinate_encoding(x) if self.coordinate_encoding else x

        # 4. K, V Projection 
        k = self.w_k(x) 
        v = self.w_v(x)

        # 5. Sampling + Similarity Calculation + Aggregation 
        if self.sampling_type == "all":
            # Q Projection 
            q = self.w_q(x)

            similarity_matrix = self._calculate_euclidean_matrix(k, q) if self.magnitude_type == 'euclidean' else self._calculate_cosine_matrix(k, q)
            prime = self._prime(v, similarity_matrix, self.K, self.maximum)

        elif self.sampling_type == "random":
            rand_idx = torch.randperm(x.shape[-1], device=x.device)[:self.num_samples]
            x_sample = x[:, :, rand_idx]

            # Q Projection 
            q = self.w_q(x_sample) 

            similarity_matrix = self._calculate_euclidean_matrix_N(k, q) if self.magnitude_type == 'euclidean' else self._calculate_cosine_matrix_N(k, q)

            range_idx = torch.arange(len(rand_idx), device=x.device)
            similarity_matrix[:, rand_idx, range_idx] = self.INF if self.magnitude_type == 'euclidean' else self.NEG_INF

            prime = self._prime_N(x, similarity_matrix, self.K, rand_idx, self.maximum)

        elif self.sampling_type == "spatial":
            spat_idx = torch.linspace(0 + self.sample_padding, x.shape[-1] - self.sample_padding - 1, self.num_samples, device=x.device).to(torch.long)
            x_sample = x[:, :, spat_idx]

            # Q Projection 
            q = self.w_q(x_sample)

            similarity_matrix = self._calculate_euclidean_matrix_N(k, q) if self.magnitude_type == 'euclidean' else self._calculate_cosine_matrix_N(k, q)

            range_idx = torch.arange(len(spat_idx), device=x.device)
            similarity_matrix[:, spat_idx, range_idx] = self.INF if self.magnitude_type == 'euclidean' else self.NEG_INF

            prime = self._prime_N(x, similarity_matrix, self.K, spat_idx, self.maximum)
        else: 
            raise ValueError("Invalid sampling_type. Must be one of ['all', 'random', 'spatial'].")

        x = self.conv1d_layer(prime)

        # 7. Pixel Shuffle
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
        torch.diagonal(similarity_matrix, dim1=1, dim2=2).fill_(1.1)
        return similarity_matrix 
    
    def _calculate_cosine_matrix_N(self, K, Q):
        norm_k = F.normalize(K, p=2, dim=1)
        norm_q = F.normalize(Q, p=2, dim=1)
        similarity_matrix = torch.bmm(norm_k.transpose(1, 2), norm_q)
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
            prime = prime.view(b, c, self.padded_shape[-1], K)
            prime = prime[:, :, self.padding:-self.padding, :]
            prime = prime.reshape(b, c, K * self.og_shape[-1])
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
        final_indices = torch.cat([token_indices, mapped_tensor], dim=-1)
        indices_expanded = final_indices.unsqueeze(1).expand(b, c, t, K)

        topk_values_exp = topk_values.unsqueeze(1).expand(b, c, t, K - 1)
        ones = torch.ones((b, c, t, 1), device=v.device)
        topk_values_exp = torch.cat([ones, topk_values_exp], dim=-1)

        # Gather matrix values and apply similarity weighting
        v_expanded = v.unsqueeze(-1).expand(b, c, t, K).contiguous()
        prime = torch.gather(v_expanded, dim=2, index=indices_expanded)
        prime = prime * topk_values_exp

        if self.padding > 0:
            prime = prime.view(b, c, self.padded_shape[-1], K)
            prime = prime[:, :, self.padding:-self.padding, :]
            prime = prime.reshape(b, c, K * self.og_shape[-1])
        else:
            prime = prime.view(b, c, -1)
        return prime

    def _add_coordinate_encoding(self, x):
        b, c, t = x.shape 
        cache_key = f"{b}_{t}_{x.device}"
        if cache_key in self.coordinate_cache: 
            expanded_coords = self.coordinate_cache[cache_key]
        else: 
            coords_vec = torch.linspace(start=-1, end=1, steps=t, device=x.device).unsqueeze(0).expand(b, -1) 
            expanded_coords = coords_vec.unsqueeze(1).expand(b, -1, -1) 
            self.coordinate_cache[cache_key] = expanded_coords

        x_with_coords = torch.cat([x, expanded_coords], dim=1) 
        return x_with_coords


class Conv1d_Branching(nn.Module):
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
        super(Conv1d_Branching, self).__init__()

        self.branch_ratio = branch_ratio
        self.in_channels = in_channels
        # self.in_channels_1 = int(in_channels * branch_ratio)
        # self.in_channels_2 = in_channels - self.in_channels_1
        self.out_channels_1 = int(out_channels * branch_ratio)
        self.out_channels_2 = out_channels - self.out_channels_1

        if self.branch_ratio != 0:
            self.branch1 = Conv1d_NN(
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
            self.branch2 = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels_2,
                kernel_size=kernel_size,
                stride=1,
                padding="same"
            )
            
        self.pointwise_conv = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )
        # self.channel_shuffle = ChannelShuffle1D(groups=2) # Optional Channel Shuffle - not in use 

    def forward(self, x):
        if self.branch_ratio == 0:
            x = self.branch2(x)
            x = self.pointwise_conv(x)
            return x

        if self.branch_ratio == 1:
            x = self.branch1(x)
            x = self.pointwise_conv(x)
            return x

    
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        # x1 = self.branch1(x[:, :self.in_channels_1, :])
        # x2 = self.branch2(x[:, self.in_channels_2:, :])
        out = torch.cat([x1, x2], dim=1)
        out = self.pointwise_conv(out)
        return out
