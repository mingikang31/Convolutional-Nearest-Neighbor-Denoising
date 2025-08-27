"""Convolutional Nearest Neighbor Layers 1D"""

"""
Layers 1D: 
(1) Conv1d_NN (All, Random, Spatial Sampling) 
(2) Conv1d_NN_Attn (All, Random, Spatial Sampling) 
(3) Attention1d 

(*) PixelShuffle1D
(*) PixelUnshuffle1D
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
class Conv1d_New(nn.Module): 
    """Convolutional Nearest Neighbor Layer 1D"""
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size,
                 stride, 
                 shuffle_pattern, 
                 shuffle_scale, 
                 coordinate_encoding
                 ):
       
        super(Conv1d_New, self).__init__()
    

        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale

        # Positional Encoding
        self.coordinate_cache = {}  # Cache for coordinate encoding
        self.coordinate_encoding = coordinate_encoding
        self.in_channels = in_channels + 1 if self.coordinate_encoding else in_channels  # Add 1 for coordinate encoding
        self.out_channels = out_channels + 1 if self.coordinate_encoding else out_channels  # Add 1 for coordinate encoding

        # Shuffle1D/Unshuffle1D Layer
        self.shuffle_layer = PixelShuffle1D(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = PixelUnshuffle1D(downscale_factor=self.shuffle_scale)

        # Adjust Channels for PixelShuffle
        self.in_channels = self.in_channels * shuffle_scale if self.shuffle_pattern in ["BA", "B"] else self.in_channels
        self.out_channels = self.out_channels * shuffle_scale if self.shuffle_pattern in ["BA", "A"] else self.out_channels

        # Conv1d Layer
        self.conv1d_layer = nn.Conv1d(in_channels=self.in_channels, 
                                      out_channels=self.out_channels, 
                                      kernel_size=self.kernel_size, 
                                      stride=self.stride, 
                                      padding="same")
        self.out_channels = self.out_channels // self.shuffle_scale if self.shuffle_pattern in ["BA", "A"] else self.out_channels

        self.pointwise_conv = nn.Conv1d(in_channels=self.out_channels, 
                                        out_channels=self.out_channels - 1, 
                                        kernel_size=1, 
                                        stride=1, 
                                        padding=0) 

    def forward(self, x):
        x = self._add_coordinate_encoding(x) if self.coordinate_encoding else x
        x = self.unshuffle_layer(x) if self.shuffle_pattern in ["B", "BA"] else x

        # Conv1d Layer
        x = self.conv1d_layer(x)
        x = self.shuffle_layer(x) if self.shuffle_pattern in ["A", "BA"] else x
        x = self.pointwise_conv(x) if self.coordinate_encoding else x  
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
        super(Conv1d_NN, self).__init__()
        
        # Assertions 
        assert K == stride, "Error: K must be same as stride. K == stride."
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert sampling_type in ["all", "random", "spatial"], "Error: sampling_type must be one of ['all', 'random', 'spatial']"
        assert num_samples > 0 or num_samples == -1, "Error: num_samples must be greater than 0 or -1 for all samples"
        assert (sampling_type == "all" and num_samples == -1) or (sampling_type != "all" and isinstance(num_samples, int)), "Error: num_samples must be -1 for 'all' sampling or an integer for 'random' and 'spatial' sampling"

        # Initialize parameters

        self.K = K
        self.stride = stride
        self.sampling_type = sampling_type
        self.num_samples = num_samples if num_samples != -1 else 'all'  # -1 for all samples
        self.sample_padding = sample_padding if sampling_type == "spatial" else 0
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.magnitude_type = magnitude_type
        self.maximum = True if self.magnitude_type == 'similarity' else False

        # Positional Encoding
        self.coordinate_cache = {}  # Cache for coordinate encoding
        self.coordinate_encoding = coordinate_encoding
        self.in_channels = in_channels + 1 if self.coordinate_encoding else in_channels  # Add 1 for coordinate encoding
        self.out_channels = out_channels + 1 if self.coordinate_encoding else out_channels  # Add 1 for coordinate encoding

        # Shuffle1D/Unshuffle1D Layer
        self.shuffle_layer = PixelShuffle1D(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = PixelUnshuffle1D(downscale_factor=self.shuffle_scale)

        # Adjust Channels for PixelShuffle
        self.in_channels = self.in_channels * shuffle_scale if self.shuffle_pattern in ["BA", "B"] else self.in_channels
        self.out_channels = self.out_channels * shuffle_scale if self.shuffle_pattern in ["BA", "A"] else self.out_channels

        # Conv1d Layer
        self.conv1d_layer = nn.Conv1d(in_channels=self.in_channels, 
                                      out_channels=self.out_channels, 
                                      kernel_size=self.K, 
                                      stride=self.stride, 
                                      padding=0)
        self.out_channels = self.out_channels // self.shuffle_scale if self.shuffle_pattern in ["BA", "A"] else self.out_channels

        self.pointwise_conv = nn.Conv1d(in_channels=self.out_channels, 
                                        out_channels=self.out_channels - 1, 
                                        kernel_size=1, 
                                        stride=1, 
                                        padding=0) 

    def forward(self, x):
        x = self._add_coordinate_encoding(x) if self.coordinate_encoding else x
        x = self.unshuffle_layer(x) if self.shuffle_pattern in ["B", "BA"] else x
        
        if self.sampling_type == "all": 
            # ConvNN Algorithm 
            matrix_magnitude = self._calculate_distance_matrix(x) if self.magnitude_type == 'distance' else self._calculate_similarity_matrix(x)
            prime = self._prime(x, matrix_magnitude, self.K, self.maximum)
        
        elif self.sampling_type == "random": 
            # Select random samples
            rand_idx = torch.randperm(x.shape[2], device=x.device)[:self.num_samples]
            x_sample = x[:, :, rand_idx]

            # ConvNN Algorithm
            matrix_magnitude = self._calculate_distance_matrix_N(x, x_sample) if self.magnitude_type == 'distance' else self._calculate_similarity_matrix_N(x, x_sample)
            range_idx = torch.arange(len(rand_idx), device=x.device)
            matrix_magnitude[:, rand_idx, range_idx] = float('inf') if self.magnitude_type == 'distance' else float('-inf')
            prime = self._prime_N(x, matrix_magnitude, self.K, rand_idx, self.maximum)


        elif self.sampling_type == "spatial":
            # Get spatial sampled indices
            spat_idx = torch.linspace(0 + self.sample_padding, x.shape[2] - self.sample_padding - 1, self.num_samples, device=x.device).to(torch.long)
            x_sample = x[:, :, spat_idx]

            # ConvNN Algorithm
            matrix_magnitude = self._calculate_distance_matrix_N(x, x_sample) if self.magnitude_type == 'distance' else self._calculate_similarity_matrix_N(x, x_sample)
            range_idx = torch.arange(len(spat_idx), device=x.device)
            matrix_magnitude[:, spat_idx, range_idx] = float('inf') if self.magnitude_type == 'distance' else float('-inf')
            prime = self._prime_N(x, matrix_magnitude, self.K, spat_idx, self.maximum)

        else: 
            raise ValueError("Invalid sampling_type. Must be one of ['all', 'random', 'spatial'].")

        x = self.conv1d_layer(prime)
        x = self.shuffle_layer(x) if self.shuffle_pattern in ["A", "BA"] else x
        x = self.pointwise_conv(x) if self.coordinate_encoding else x  
        return x
    
    def _calculate_distance_matrix(self, matrix, sqrt=False):
        norm_squared = torch.sum(matrix ** 2, dim=1, keepdim=True)
        dot_product = torch.bmm(matrix.transpose(2, 1), matrix)
        
        dist_matrix = norm_squared + norm_squared.transpose(2, 1) - 2 * dot_product
        dist_matrix = torch.clamp(dist_matrix, min=0) # remove negative values
        dist_matrix = torch.sqrt(dist_matrix) if sqrt else dist_matrix # take square root if needed
        
        return dist_matrix
    
    def _calculate_distance_matrix_N(self, matrix, matrix_sample, sqrt=False):
        norm_squared = torch.sum(matrix ** 2, dim=1, keepdim=True).permute(0, 2, 1)
        norm_squared_sample = torch.sum(matrix_sample ** 2, dim=1, keepdim=True).transpose(2, 1).permute(0, 2, 1)
        dot_product = torch.bmm(matrix.transpose(2, 1), matrix_sample)
        
        dist_matrix = norm_squared + norm_squared_sample - 2 * dot_product
        dist_matrix = torch.clamp(dist_matrix, min=0) # remove negative values
        dist_matrix = torch.sqrt(dist_matrix) if sqrt else dist_matrix

        return dist_matrix
    
    def _calculate_similarity_matrix(self, matrix):
        # p=2 (L2 Norm - Euclidean Distance), dim=1 (across the channels)
        norm_matrix = F.normalize(matrix, p=2, dim=1) 
        similarity_matrix = torch.bmm(norm_matrix.transpose(2, 1), norm_matrix)
        return similarity_matrix
    
    def _calculate_similarity_matrix_N(self, matrix, matrix_sample):
        # p=2 (L2 Norm - Euclidean Distance), dim=1 (across the channels)
        norm_matrix = F.normalize(matrix, p=2, dim=1) 
        norm_sample = F.normalize(matrix_sample, p=2, dim=1)
        similarity_matrix = torch.bmm(norm_matrix.transpose(2, 1), norm_sample)
        return similarity_matrix

    def _prime(self, matrix, magnitude_matrix, K, maximum):
        b, c, t = matrix.shape
        _, topk_indices = torch.topk(magnitude_matrix, k=K, dim=2, largest=maximum)
        topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)    
      
        matrix_expanded = matrix.unsqueeze(-1).expand(b, c, t, K).contiguous()
        prime = torch.gather(matrix_expanded, dim=2, index=topk_indices_exp)
        prime = prime.view(b, c, -1)
        return prime
    
    def _prime_N(self, matrix, magnitude_matrix, K, rand_idx, maximum):
        b, c, t = matrix.shape
        _, topk_indices = torch.topk(magnitude_matrix, k=K - 1, dim=2, largest=maximum)
        tk = topk_indices.shape[-1]
        assert K == tk + 1, "Error: K must be same as tk + 1. K == tk + 1."

        mapped_tensor = rand_idx[topk_indices]
        token_indices = torch.arange(t, device=matrix.device).view(1, t, 1).expand(b, t, 1)
        final_indices = torch.cat([token_indices, mapped_tensor], dim=2)
        indices_expanded = final_indices.unsqueeze(1).expand(b, c, t, K)

        matrix_expanded = matrix.unsqueeze(-1).expand(b, c, t, K).contiguous()
        prime = torch.gather(matrix_expanded, dim=2, index=indices_expanded)  
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
    """Convolutional Nearest Neighbor Layer 1D with Attention"""
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
                 num_tokens, 
                 magnitude_type, 
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
            num_tokens (int): Number of tokens for attention.   
            magnitude_type (str): Distance or Similarity.
        """
        super(Conv1d_NN_Attn, self).__init__()
        
        # Assertions 
        assert K == stride, "Error: K must be same as stride. K == stride."
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert magnitude_type in ["distance", "similarity"], "Error: magnitude_type must be one of ['distance', 'similarity']"
        assert sampling_type in ["all", "random", "spatial"], "Error: sampling_type must be one of ['all', 'random', 'spatial']"
        assert num_samples > 0 or num_samples == -1, "Error: num_samples must be greater than 0 or -1 for all samples"
        assert isinstance(num_tokens, int) and num_tokens > 0, "Error: num_tokens must be a positive integer"
        assert (sampling_type == "all" and num_samples == -1) or (sampling_type != "all" and isinstance(num_samples, int)), "Error: num_samples must be -1 for 'all' sampling or an integer for 'random' and 'spatial' sampling"

        # Initialize parameters

        self.K = K
        self.stride = stride
        self.sampling_type = sampling_type
        self.num_samples = num_samples if num_samples != -1 else 'all'  # -1 for all samples
        self.sample_padding = sample_padding if sampling_type == "spatial" else 0
        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        self.num_tokens = int(num_tokens/2) if self.shuffle_pattern in ["B", "BA"] else num_tokens
        self.magnitude_type = magnitude_type
        self.maximum = True if self.magnitude_type == 'similarity' else False

        # Positional Encoding
        self.coordinate_cache = {}  # Cache for coordinate encoding
        self.coordinate_encoding = coordinate_encoding
        self.in_channels = in_channels + 1 if self.coordinate_encoding else in_channels  # Add 1 for coordinate encoding
        self.out_channels = out_channels + 1 if self.coordinate_encoding else out_channels  # Add 1 for coordinate encoding
        
        # Shuffle1D/Unshuffle1D Layer
        self.shuffle_layer = PixelShuffle1D(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = PixelUnshuffle1D(downscale_factor=self.shuffle_scale)

        # Adjust Channels for PixelShuffle
        self.in_channels = self.in_channels * shuffle_scale if self.shuffle_pattern in ["BA", "B"] else self.in_channels
        self.out_channels = self.out_channels * shuffle_scale if self.shuffle_pattern in ["BA", "A"] else self.out_channels

        # Conv1d Layer
        self.conv1d_layer = nn.Conv1d(in_channels=self.in_channels, 
                                      out_channels=self.out_channels, 
                                      kernel_size=self.K, 
                                      stride=self.stride, 
                                      padding=0)

        # Linear Projections for Q, K, V, O
        self.w_q = nn.Linear(self.num_tokens, self.num_tokens, bias=False) if self.sampling_type == "all" else nn.Linear(self.num_samples, self.num_samples, bias=False)
        self.w_k = nn.Linear(self.num_tokens, self.num_tokens, bias=False) 
        self.w_v = nn.Linear(self.num_tokens, self.num_tokens, bias=False) 
        self.w_o = nn.Linear(self.num_tokens, self.num_tokens, bias=False)
        
        self.attention_dropout = attention_dropout
        self.dropout = nn.Dropout(p=self.attention_dropout)

        self.out_channels = self.out_channels // self.shuffle_scale if self.shuffle_pattern in ["BA", "A"] else self.out_channels

        # Pointwise Conv1d Layer
        self.pointwise_conv = nn.Conv1d(in_channels=self.out_channels, 
                                        out_channels=self.out_channels - 1, 
                                        kernel_size=1, 
                                        stride=1, 
                                        padding=0)

    def forward(self, x):
        # Unshuffle 
        x = self._add_coordinate_encoding(x) if self.coordinate_encoding else x
        x = self.unshuffle_layer(x) if self.shuffle_pattern in ["B", "BA"] else x

        # K, V Projections 
        k = self.w_k(x)
        v = self.w_v(x)
        
        if self.sampling_type == "all": 
            # Q Projection
            q = self.w_q(x)
            
            # ConvNN Algorithm
            matrix_magnitude = self._calculate_distance_matrix(k, q, sqrt=True) if self.magnitude_type == 'distance' else self._calculate_similarity_matrix(k, q)
            prime = self._prime(v, matrix_magnitude, self.K, self.maximum)
        
        elif self.sampling_type == "random": 
            # Select random samples
            rand_idx = torch.randperm(x.shape[2], device=x.device)[:self.num_samples]
            x_sample = x[:, :, rand_idx]

            # Q Projection
            q = self.w_q(x_sample)

            # ConvNN Algorithm 
            matrix_magnitude = self._calculate_distance_matrix_N(k, q, sqrt=True) if self.magnitude_type == 'distance' else self._calculate_similarity_matrix_N(k, q)
            range_idx = torch.arange(len(rand_idx), device=x.device)
            matrix_magnitude[:, rand_idx, range_idx] = float('inf') if self.magnitude_type == 'distance' else float('-inf')
            prime = self._prime_N(v, matrix_magnitude, self.K, rand_idx, self.maximum)

        elif self.sampling_type == "spatial":
            # Get spatial sampled indices
            spat_idx = torch.linspace(0 + self.sample_padding, x.shape[2] - self.sample_padding - 1, self.num_samples, device=x.device).to(torch.long)
            x_sample = x[:, :, spat_idx]

            # Q Projection
            q = self.w_q(x_sample)

            # ConvNN Algorithm 
            matrix_magnitude = self._calculate_distance_matrix_N(k, q, sqrt=True) if self.magnitude_type == 'distance' else self._calculate_similarity_matrix_N(k, q)
            range_idx = torch.arange(len(spat_idx), device=x.device)
            matrix_magnitude[:, spat_idx, range_idx] = float('inf') if self.magnitude_type == 'distance' else float('-inf')
            prime = self._prime_N(v, matrix_magnitude, self.K, spat_idx, self.maximum)

        else: 
            raise ValueError("Invalid sampling_type. Must be one of ['all', 'random', 'spatial'].")
        
        x = self.conv1d_layer(prime)
        x = self.w_o(x)
        x = self.shuffle_layer(x) if self.shuffle_pattern in ["A", "BA"] else x
        x = self.pointwise_conv(x) if self.coordinate_encoding else x
        return x 

    def _calculate_similarity_matrix(self, K, Q):
        k_norm = F.normalize(K, p=2, dim=1)
        q_norm = F.normalize(Q, p=2, dim=1)
        similarity_matrix = torch.bmm(k_norm.transpose(2, 1), q_norm) 
        similarity_matrix = torch.clamp(similarity_matrix, min=0)  
        return similarity_matrix
    
    def _calculate_similarity_matrix_N(self, K, Q):
        k_norm = F.normalize(K, p=2, dim=1)
        q_norm = F.normalize(Q, p=2, dim=1)
        similarity_matrix = torch.bmm(k_norm.transpose(2, 1), q_norm)  
        similarity_matrix = torch.clamp(similarity_matrix, min=0)
        return similarity_matrix
        
    def _calculate_distance_matrix(self, K, Q, sqrt=False):
        norm_squared_K = torch.sum(K**2, dim=1, keepdim=True) 
        norm_squared_Q = torch.sum(Q**2, dim=1, keepdim=True) 
        dot_product = torch.bmm(K.transpose(2, 1), Q)  
        dist_matrix = norm_squared_K + norm_squared_Q.transpose(2, 1) - 2 * dot_product
        dist_matrix = torch.clamp(dist_matrix, min=0)  # remove negative values
        dist_matrix = torch.sqrt(dist_matrix) if sqrt else dist_matrix  # take square root if needed
        return dist_matrix

    def _calculate_distance_matrix_N(self, K, Q, sqrt=False):
        norm_squared_K = torch.sum(K**2, dim=1, keepdim=True).permute(0, 2, 1)
        norm_squared_Q = torch.sum(Q**2, dim=1, keepdim=True).transpose(2, 1).permute(0, 2, 1)
        dot_product = torch.bmm(K.transpose(2, 1), Q)  
        dist_matrix = norm_squared_K + norm_squared_Q - 2 * dot_product
        dist_matrix = torch.clamp(dist_matrix, min=0)  # remove negative values
        dist_matrix = torch.sqrt(dist_matrix) if sqrt else dist_matrix  # take square root if needed
        return dist_matrix

    def _prime(self, v, qk, K, maximum):
        b, c, t = v.shape 
        _, topk_indices = torch.topk(qk, k=K, dim=-1, largest = maximum)
        topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)
        
        v_expanded = v.unsqueeze(-1).expand(b, c, t, K)
        prime = torch.gather(v_expanded, dim=2, index=topk_indices_exp)
        prime = prime.reshape(b, c, -1)
        return prime
            
    def _prime_N(self, v, qk, K, rand_idx, maximum):
        b, c, t = v.shape
        _, topk_indices = torch.topk(qk, k=K - 1, dim=2, largest=maximum)
        tk = topk_indices.shape[-1]
        assert K == tk + 1, "Error: K must be same as tk + 1. K == tk + 1."
        mapped_tensor = rand_idx[topk_indices]

        token_indices = torch.arange(t, device=v.device).view(1, t, 1).expand(b, t, 1)
        final_indices = torch.cat([token_indices, mapped_tensor], dim=2)
        indices_expanded = final_indices.unsqueeze(1).expand(b, c, t, K)

        v_expanded = v.unsqueeze(-1).expand(b, c, t, K).contiguous()
        prime = torch.gather(v_expanded, dim=2, index=indices_expanded) 
        prime = prime.reshape(b, c, -1)
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


"""(3) Attention1d"""
class Attention1d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_heads,
                 shuffle_pattern,
                 shuffle_scale 
                 ):
        super(Attention1d, self).__init__()
        
        assert shuffle_pattern in ["B", "A", "BA", "NA"], "Error: shuffle_pattern must be one of ['B', 'A', 'BA', 'NA']"
        assert shuffle_scale > 0, "Error: shuffle_scale must be greater than 0"
        
        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        self.shuffle_pattern = shuffle_pattern
        self.shuffle_scale = shuffle_scale
        
        # Shuffle1D/Unshuffle1D Layer
        self.shuffle_layer = PixelShuffle1D(upscale_factor=self.shuffle_scale)
        self.unshuffle_layer = PixelUnshuffle1D(downscale_factor=self.shuffle_scale)
        
        # Adjust Channels for Shuffle
        self.in_channels = self.in_channels * self.shuffle_scale if self.shuffle_pattern in ["BA", "B"] else in_channels
        self.out_channels = self.out_channels * self.shuffle_scale if self.shuffle_pattern in ["BA", "A"] else out_channels
        
        # MultiHead Attention Layer
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=self.out_channels, num_heads=self.num_heads, batch_first=True)
        
        # 1x1 Convolution Layer
        self.conv1x1 = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1)
        
    def forward(self, x):
        if self.shuffle_pattern in ["BA", "B"]:
            x1 = self.unshuffle_layer(x)
        else: 
            x1 = x 
        
        x1 = self.conv1x1(x1) # [B, C, N]
        x1 = x1.permute(0, 2, 1)
        
        x2 = self.multi_head_attention(x1, x1, x1)[0] # (B, N, C)
        x2 = x2.permute(0, 2, 1) # (B, C, N)
        
        if self.shuffle_pattern in ["BA", "A"]:
            x3 = self.shuffle_layer(x2)
        else: 
            x3 = x2
        return x3

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