"""
Implementation of the language-enhanced transformer architecture (LEA).
Code adapted from minGPT (https://github.com/karpathy/minGPT).
"""
#----------------------------------------------------
# import libraries
#----------------------------------------------------
import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, reduce, repeat
#----------------------------------------------------

#----------------------------------------------------
# class definitions
#----------------------------------------------------
class GPTConfig:
    """
    Configuration class to store the configuration of a `GPTFactor`. 
    """
    embd_pdrop = 0.
    resid_pdrop = 0.
    attn_pdrop = 0.

    def __init__(self, vocab_size, block_size, **kwargs):

        """
        Set the parameters of the model.

        Args:
            vocab_size: size of the token vocabulary
            block_size: size of the token sequence
            Additional arguments:
                - embd_pdrop (float): dropout probability for the embedding layer
                - resid_pdrop (float): dropout probability for the residual layer
                - attn_pdrop (float): dropout probability for the attention layer
                - n_layer (int): number of transformer blocks
                - n_head (int): number of attention heads
                - n_embd (int): embedding dimension
                - n_embd_pos (int): embedding dimension for the position encodings
                - n_unmasked (int): number of tokens to leave unmasked in the causal mask (= block_size), kept for legacy reasons              
                - vocab_speech_size (int): size of the speech token vocabulary
                - block_speech_size (int): length of a sentence
                - vq_type (str): type of vector quantization to use (vanilla or EMA)
                - bool_blank_speech (bool): whether to use blank speech or not
        Returns:
            None
        """

        self.vocab_size = vocab_size
        self.block_size = block_size # max content size in tokens
        for k, v in kwargs.items():
            setattr(self, k, v)

#....................................................
            
class SelfAttention(nn.Module):

    """
    A vanilla multi-head self-attention block with linear projection at the end.
    """

    def __init__(self, config):

        """
        Create a SelfAttention object.

        Args:
            config (GPTConfig): configuration of the model
        Returns:
            None
        """

        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head

    def forward(self, x):

        """
        Forward pass of the self-attention block.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_length, n_embd)
        Returns:
            y (torch.Tensor): output tensor of shape (batch_size, seq_length, n_embd)
        """
        B, T, C = x.size() # T = sequence length, C = embedding dim

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
    
class CrossAttention(nn.Module):

    """
    A vanilla multi-head cross-attention block with linear projection at the end.
    """

    def __init__(self, config):

        """
        Create a CrossAttention object.

        Args:
            config (GPTConfig): configuration of the model
        Returns:
            None
        """

        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head

    def forward(self, x, z):
        """
        Forward pass of the cross-attention block.

        Abs:
            x (torch.Tensor): input tensor of shape (batch_size, seq_length, n_embd), used to produce the queries
            z (torch.Tensor): input tensor of shape (batch_size, seq_length, n_embd), used to produce the keys and values
        Returns:
            y (torch.Tensor): output tensor of shape (batch_size, seq_length, n_embd)
        """
        B, Tx, C = x.size() # T = sequence length, C = embedding dim
        B, Tz, C = z.size() # T = sequence length, C = embedding dim

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, Tx, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(z).view(B, Tz, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(z).view(B, Tz, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, Tx, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
    
class BlockDecoder(nn.Module):
    """
    Transformer block decoder featuring self-attention, cross-attention, and MLP.
    """

    def __init__(self, config):

        """
        Create a BlockDecoder object.

        Args:
            config (GPTConfig): configuration of the model
        Returns:
            None
        """
        super().__init__()
        self.ln_x_1 = nn.LayerNorm(config.n_embd)
        self.ln_x_2 = nn.LayerNorm(config.n_embd)
        self.ln_z = nn.LayerNorm(config.n_embd)
        self.ln_mlp = nn.LayerNorm(config.n_embd)

        self.selfAttn = SelfAttention(config)
        self.crossAttn = CrossAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, z, bool_selfAttn_first = True):

        """
        Forward pass of the transformer block decoder.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_length, n_embd)
            z (torch.Tensor): input tensor of shape (batch_size, seq_length, n_embd)
            bool_selfAttn_first (bool): whether to perform self-attention first or cross-attention first
        Returns:
            y (torch.Tensor): output tensor of shape (batch_size, seq_length, n_embd)
        """

        if bool_selfAttn_first: # self attention first
            x = x + self.selfAttn(self.ln_x_1(x))
            x = x + self.crossAttn(self.ln_x_2(x), self.ln_z(z))
        
        else: # cross attention first
            x = x + self.crossAttn(self.ln_x_2(x), self.ln_z(z))
            x = x + self.selfAttn(self.ln_x_1(x))

        x = x + self.mlp(self.ln_mlp(x))
        return x

#....................................................

class SelfAttentionFactor(nn.Module):

    """
    Multi-head self-attention block with linear projection at the end. Input dimensions allow for factorized token/positional embeddings.
    """

    def __init__(self, config):

        """
        Create a SelfAttentionFactor object.

        Args:
            config (GPTConfig): configuration of the model
        Returns:
            None
        """

        super().__init__()
        assert (config.n_embd + config.n_embd_pos) % config.n_head == 0
        self.Din = config.n_embd + config.n_embd_pos
        self.Dout = config.n_embd
        # key, query, value projections for all heads
        self.key = nn.Linear(self.Din, config.n_embd)
        self.query = nn.Linear(self.Din, config.n_embd)
        self.value = nn.Linear(self.Din, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head

    def forward(self, x):

        """
        Forward pass of the self-attention block.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_length, n_embd + n_embd_pos)
        Returns:
            y (torch.Tensor): output tensor of shape (batch_size, seq_length, n_embd)
        """

        B, T, _ = x.size() # T = sequence length, C = embedding dim

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, self.Dout // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, self.Dout // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, self.Dout // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, self.Dout)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
    
class CrossAttentionFactor(nn.Module):

    """
    Multi-head cross-attention block with linear projection at the end. Input dimensions allow for factorized token/positional embeddings.
    """

    def __init__(self, config):

        """
        Create a CrossAttentionFactor object.

        Args:
            config (GPTConfig): configuration of the model
        Returns:
            None
        """

        super().__init__()
        assert (config.n_embd + config.n_embd_pos) % config.n_head == 0
        self.Din = config.n_embd + config.n_embd_pos
        self.Dout = config.n_embd
        # key, query, value projections for all heads
        self.key = nn.Linear(self.Din, config.n_embd)
        self.query = nn.Linear(self.Din, config.n_embd)
        self.value = nn.Linear(self.Din, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head

    def forward(self, x, z):
        
        """
        Forward pass of the cross-attention block.

        Abs:
            x (torch.Tensor): input tensor of shape (batch_size, seq_length, n_embd + n_embd_pos), used to produce the queries
            z (torch.Tensor): input tensor of shape (batch_size, seq_length, n_embd + n_embd_pos), used to produce the keys and values
        Returns:
            y (torch.Tensor): output tensor of shape (batch_size, seq_length, n_embd)
        """

        B, Tx, _ = x.size() # T = sequence length, C = embedding dim
        B, Tz, _ = z.size() # T = sequence length, C = embedding dim

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, Tx, self.n_head, self.Dout // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(z).view(B, Tz, self.n_head, self.Dout // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(z).view(B, Tz, self.n_head, self.Dout // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, Tx, self.Dout)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
    
class BlockDecoderFactor(nn.Module):
    """
    Transformer block decoder featuring self-attention, cross-attention, and MLP. Input dimensions allow for factorized token/positional embeddings.
    """

    def __init__(self, config):

        """
        Create a BlockDecoderFactor object.

        Args:
            config (GPTConfig): configuration of the model
        Returns:
            None
        """

        super().__init__()
        self.ln_x_1 = nn.LayerNorm(config.n_embd + config.n_embd_pos)
        self.ln_x_2 = nn.LayerNorm(config.n_embd + config.n_embd_pos)
        self.ln_z = nn.LayerNorm(config.n_embd + config.n_embd_pos)
        self.ln_mlp = nn.LayerNorm(config.n_embd + config.n_embd_pos)

        self.selfAttn = SelfAttentionFactor(config)
        self.crossAttn = CrossAttentionFactor(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd + config.n_embd_pos, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, z, pos_x, pos_z, bool_selfAttn_first = True):

        """
        Forward pass of the transformer block decoder.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_length, n_embd)
            z (torch.Tensor): input tensor of shape (batch_size, seq_length, n_embd)
            pos_x (torch.Tensor): positional encoding tensor of shape (batch_size, seq_length, n_embd_pos)
            pos_z (torch.Tensor): positional encoding tensor of shape (batch_size, seq_length, n_embd_pos)
            bool_selfAttn_first (bool): whether to perform self-attention first or cross-attention first
        Returns:
            y (torch.Tensor): output tensor of shape (batch_size, seq_length, n_embd)
        """

        b, l, d = x.size()
        if pos_x.size(0) == 1:
            pos_x = repeat(pos_x, '1 n d -> b n d', b = b)
        if pos_z.size(0) == 1:
            pos_z = repeat(pos_z, '1 n d -> b n d', b = b)

        if bool_selfAttn_first: # self attention first

            xp = torch.cat((x, pos_x), dim = 2)
            x = x + self.selfAttn(self.ln_x_1(xp))
            xp = torch.cat((x, pos_x), dim = 2)
            zp = torch.cat((z, pos_z), dim = 2)
            x = x + self.crossAttn(self.ln_x_2(xp), self.ln_z(zp))
        
        else: # cross attention first
            xp = torch.cat((x, pos_x), dim = 2)
            zp = torch.cat((z, pos_z), dim = 2)
            x = x + self.crossAttn(self.ln_x_2(xp), self.ln_z(zp))
            xp = torch.cat((x, pos_x), dim = 2)
            x = x + self.selfAttn(self.ln_x_1(xp))

        xp = torch.cat((x, pos_x), dim = 2)
        x = x + self.mlp(self.ln_mlp(xp))
        return x

#....................................................

class VectorQuantizer(nn.Module):

    """
    Implementation of a vanilla vector quantizer.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost, usage_cost):
        
        """
        Creates a VectorQuantizer object.

        Args:
            num_embeddings (int): size of the codebook K
            embedding_dim (int): dimension of the embeddings
            commitment_cost (float): commitment cost
            usage_cost (float): usage cost
        Returns:
            None
        """
        
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim # dimension of embedding D
        self._num_embeddings = num_embeddings # size of the codebook K
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim) #(K,D)
        self._embedding.weight.data.normal_() #init from N(0,1)
        self._commitment_cost = commitment_cost
        self._usage_cost = usage_cost

    def forward(self, x, device = 'cpu'):
        
        """
        Forward pass of the vector quantizer.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_length, embedding_dim)
            device (str): device to be used
        Returns:
            quantized (torch.Tensor): quantized tensor of shape (batch_size, seq_length, embedding_dim)
            loss (torch.Tensor): loss tensor of shape (1,)
            encodings (torch.Tensor): one-hot encodings tensor of shape (batch_size, seq_length, num_embeddings)
            encoding_indices (torch.Tensor): encoding indices tensor of shape (batch_size, seq_length)
        """
        
        b, l, d = x.size()
        
        # Flatten input
        flat_x = rearrange(x, 'b l d -> (b l) d') 
        
        # Calculate distances or vector to quanta in codebook (L2(x,y) = (x-y)^2 = x^2+y^2-2*x*y)
        distances = torch.cdist(flat_x, self._embedding.weight)
            
        # Encode
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, num_classes=self._num_embeddings).double()
        quantized = self._embedding(encoding_indices)
        
        # Reshape
        quantized = rearrange(quantized, '(b l) d -> b l d', b = b, l = l, d = d)
        encoding_indices = rearrange(encoding_indices, '(b l) -> b l', b = b, l = l)

        # Compute loss
        e_latent_loss = F.mse_loss(quantized.detach(), x) # get inputs closer to quanta
        q_latent_loss = F.mse_loss(quantized, x.detach()) # get quanta closer to inputs
        
        # Calculate usage loss
        usage = torch.mean(encodings, dim=0)  # Average usage across the batch
        usage_loss = ((usage - 1/self._num_embeddings) ** 2).mean()  # Encourage uniform usage
        
        loss = q_latent_loss + self._commitment_cost * e_latent_loss + self._usage_cost * usage_loss

        # reshape encodings
        encodings = rearrange(encodings, '(b l) v -> b l v', b = b, l = l)  

        # transfer gradients in inputs to quantized
        quantized = x + (quantized - x).detach()

        return quantized, loss, encodings, encoding_indices
    
    def get_codebook_entries(self, indices):
        """
        get codebook entry for indices

        Args:
            indices (torch.Tensor): tensor of indices
        Returns:
            codebook_entries (torch.Tensor): tensor of embeddings
        """
        return self._embedding(indices)
    
    def _quantize(self, x):
        pass

class VectorQuantizerEMA(nn.Module):

    """
    Implementation of a vector quantizer with exponential moving averages (EMA).
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost, usage_cost, decay = 0.99, epsilon=1e-20, xavier_init = False):
        
        """
        Creates a VectorQuantizerEMA object.

        Args:
            num_embeddings (int): size of the codebook K
            embedding_dim (int): dimension of the embeddings
            commitment_cost (float): commitment cost
            usage_cost (float): usage cost
            decay (float): decay for the EMA
            epsilon (float): epsilon for the EMA
            xavier_init (bool): whether to use Xavier initialization or not
        Returns:
            None
        """
        
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings # vocabular size
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        if xavier_init:
            nn.init.xavier_uniform_(self.embedding.weight.data)
        else:
            self._embedding.weight.data.normal_()

        self._commitment_cost = commitment_cost
        self._usage_cost = usage_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        
        self._ema_w = nn.Parameter(torch.Tensor(self._num_embeddings, self._embedding_dim))
        if xavier_init:
            nn.init.xavier_uniform_(self._ema_w.data)
        else:
            self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, x):

        """
        Forward pass of the vector quantizer with EMA.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_length, embedding_dim)
        Returns:
            quantized (torch.Tensor): quantized tensor of shape (batch_size, seq_length, embedding_dim)
            loss (torch.Tensor): loss tensor of shape (1,)
            encodings (torch.Tensor): one-hot encodings tensor of shape (batch_size, seq_length, num_embeddings)
            encoding_indices (torch.Tensor): encoding indices tensor of shape (batch_size, seq_length)
        """

        b, l, d = x.size()
        # Flatten input
        flat_x = rearrange(x, 'b l d -> (b l) d') 
        
        # Calculate distances
        distances = torch.cdist(flat_x, self._embedding.weight)
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=-1)
        encodings = F.one_hot(encoding_indices, num_classes=self._num_embeddings).to(flat_x.dtype)
        quantized = rearrange(self._embedding(encoding_indices), '(b l) d -> b l d', b = b, l = l)
                
        # Use EMA to update the embedding vectors
        if self.training:
            # update cluster size
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_x)
            self._ema_w.data.mul_(self._decay).add_(dw * (1 - self._decay))
            self._embedding.weight.data.copy_(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), x) # get x closer to quanta

        usage = torch.mean(encodings, dim=0)  # Average usage across the batch
        usage_loss = ((usage - 1/self._num_embeddings) ** 2).mean()  # Encourage uniform usage

        loss = self._commitment_cost * e_latent_loss + self._usage_cost * usage_loss
        
        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
        encodings = rearrange(encodings, '(b l) v -> b l v', b = b, l = l) 

        return quantized, loss, encodings, encoding_indices
    
    def get_codebook_entries(self, indices):
        """
        get codebook entry for indices

        Args:
            indices (torch.Tensor): tensor of indices
        Returns:
            codebook_entries (torch.Tensor): tensor of embeddings
        """
        return self._embedding(indices)

#....................................................

class LEA(nn.Module):

    """
    Implementation of the language-enhanced transformer architecture (LEA).
    """

    def __init__(self, vocab_size, block_size, n_layer=2, n_head=4, n_embd=64, 
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0, vocab_speech_size=50, block_speech_size = 10, vq_type = 'EMA', bool_blank_speech = True):
        
        """
        Creates a LEA object.
        
        Args:
            vocab_size (int): size of the token vocabulary
            block_size (int): size of the token sequence
            n_layer (int): number of transformer blocks
            n_head (int): number of attention heads
            n_embd (int): embedding dimension
            embd_pdrop (float): dropout probability for the embedding layer
            resid_pdrop (float): dropout probability for the residual layer
            attn_pdrop (float): dropout probability for the attention layer
            n_unmasked (int): number of tokens to leave unmasked in the causal mask (= block_size), kept for legacy reasons
            vocab_speech_size (int): size of the speech token vocabulary
            block_speech_size (int): length of a sentence
            vq_type (str): type of vector quantization to use (vanilla or EMA)
            bool_blank_speech (bool): whether to use blank speech or not

        Returns:
            None
        """

        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, # board token vocab
                           block_size=block_size,
                           n_layer=n_layer, 
                           n_head=n_head, 
                           n_embd=n_embd,
                           embd_pdrop=embd_pdrop, 
                           resid_pdrop=resid_pdrop, 
                           attn_pdrop=attn_pdrop,
                           n_unmasked=n_unmasked, # depreciated
                           vocab_speech_size=vocab_speech_size, # speech token vocab
                           block_speech_size=block_speech_size,
                           vq_type = vq_type,
                           bool_blank_speech = bool_blank_speech)
        
        #........................................
        # picture stream "ps" (masked board / speech -> reconstructed board), Inference Network (IN)
        #........................................
        
        # embeddings
        self.ps_token_embedding = nn.Embedding(config.vocab_size, config.n_embd) # token embeddings
        self.ps_pos_emb = nn.Embedding(config.block_size, config.n_embd) # position encodings
        self.ps_drop = nn.Dropout(config.embd_pdrop)

        # transformer
        # using decoder because it should cross attend to speech
        self.ps_blocks = nn.ModuleList([BlockDecoder(config) for _ in range(config.n_layer)])

        # head
        # convert representations in logits over vocabulary for each picture token
        self.ps_ln_out = nn.LayerNorm(config.n_embd)
        self.ps_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        #........................................
        # speech stream (IN latent representations -> speech), Auxiliary Language Network (ALN)
        #........................................

        # embeddings
        if vq_type == 'EMA':
            print('using VectorQuantizerEMA')
            self.ss_vector_quantizer = VectorQuantizerEMA(config.vocab_speech_size, config.n_embd, commitment_cost = 0.25, usage_cost=0.25)
        elif vq_type == 'vanilla':
            print('using VectorQuantizer')
            self.ss_vector_quantizer = VectorQuantizer(config.vocab_speech_size, config.n_embd, commitment_cost = 0.25, usage_cost=0.25)
        else:
            raise ValueError('vq_type not recognized')
        
        # what will be used as the initial token when generating speech
        self.ss_token_init = nn.Embedding(config.block_speech_size, config.n_embd) # will be learnt
        self.ss_pos_emb = nn.Embedding(config.block_speech_size, config.n_embd)  
        self.ss_drop = nn.Dropout(config.embd_pdrop)

        # transformer
        # using decoder because it should cross attend to picture stream
        self.ss_blocks = nn.ModuleList([BlockDecoder(config) for _ in range(config.n_layer)])

        # head
        # convert representations in logits over vocabulary for each picture token
        self.ss_ln_out = nn.LayerNorm(config.n_embd)

        #........................................
        # Initialize weights
        #........................................

        self.apply(self._init_weights) # init weights recursively
        self.config = config

    def _init_weights(self, module):

        """
        Initialize the weights of the model.
        """

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_idx):

        """
        Forward pass of the LEA model.

        Args:
            input_idx (torch.Tensor): input tensor of shape (batch_size, seq_length), repesents boards
        
        Returns:
            ps_logits (torch.Tensor): logits produced by IN from masked boards
            ps2_logits (torch.Tensor): logits produced by IN from speech alone (output of ALN)
            vq_loss (torch.Tensor): loss of the vector quantizer
            encodings (torch.Tensor): one-hot encodings of the speech tokens
        """
        
        device = input_idx.device
        batch_size, seq_length = input_idx.size()

        # make picutre and speech positional encodings
        position_embeddings_picture = torch.arange(0, self.config.block_size, dtype=torch.long, device=device).unsqueeze(0) # idx of position, shape (1, t)
        position_embeddings_picture = self.ps_pos_emb(position_embeddings_picture) # shape (1, t, d)

        position_embeddings_speech = torch.arange(0, self.config.block_speech_size, dtype=torch.long, device=device).unsqueeze(0) # idx of position, shape (1, t)
        position_embeddings_speech = self.ss_pos_emb(position_embeddings_speech) # shape (1, t, d)

        #-------------------------------------
        # IN: Reconstruct board from masked input
        # with blank speech (masked boards -> reconstructed boards)
        #-------------------------------------

        # initialize picture
        ps_token_embeddings = self.ps_token_embedding(input_idx)  # each index maps to a (learnable) vector
        
        # add position embeddings and token embeddings
        ps_z = ps_token_embeddings + position_embeddings_picture # broadcasting (B,block_size,n_embd)
        ps_z = self.ps_drop(ps_z)

        # initialize speech
        if self.config.bool_blank_speech:
            ps_q = torch.zeros((batch_size, self.config.block_speech_size, self.config.n_embd)).to(device)
        else:# could also use the last token in vocabulary
            ps_q_idx = (self.config.vocab_speech_size - 1) * torch.ones((batch_size, self.config.block_speech_size)).type(torch.LongTensor).to(device)
            ps_q = self.ss_vector_quantizer.get_codebook_entries(ps_q_idx)
        
        # add position embeddings
        ps_q = ps_q + position_embeddings_speech # broadcasting

        for block in self.ps_blocks:
            ps_z = block(ps_z, ps_q) # /!\ cross attention on different block sizes

        ps_z = self.ps_ln_out(ps_z)
        ps_logits = self.ps_head(ps_z) # prediction logits from picture

        #-------------------------------------
        # ALN: Construct speech from internal representations 
        # of the picture stream (IN latent representations -> speech)
        #-------------------------------------

        # initialize speech
        ss_z_idx = torch.arange(0, self.config.block_speech_size, dtype=torch.long, device=device).unsqueeze(0)
        ss_z = self.ss_token_init(ss_z_idx) # (1, block_speech_size, n_embd_speech)
        ss_z = repeat(ss_z, '1 n d -> b n d', b = batch_size)

        ss_z = ss_z + position_embeddings_speech # broadcasting (B,block_speech_size,n_embd_speech)
        ss_z = self.ss_drop(ss_z)

        for block in self.ss_blocks:
            ss_z = block(ss_z, ps_z) # /!\ cross attention on different block sizes

        ss_z = self.ss_ln_out(ss_z)

        # vector quatize to get discrete tokens
        ss_vq, vq_loss, encodings, encoding_indices = self.ss_vector_quantizer(ss_z)
        
        #-------------------------------------
        # IN: Reconstruct board from speech alone
        # (speech -> reconstructed boards)
        #------------------------------------- 

        # initialize picture
        # get token embeddings
        ps2_z_idx = (self.config.vocab_size - 1) * torch.ones((batch_size, seq_length)).type(torch.LongTensor).to(device)
        ps2_z = self.ps_token_embedding(ps2_z_idx)  # each index maps to a (learnable) vector
        
        # add position embeddings
        ps2_z = ps2_z + position_embeddings_picture # broadcasting (B,block_size,n_embd)
        ps2_z = self.ps_drop(ps2_z)

        # initialize speech
        ps2_q = ss_vq + position_embeddings_speech # broadcasting

        for block in self.ps_blocks:
            ps2_z = block(ps2_z, ps2_q) # /!\ cross attention on different block sizes

        ps2_z = self.ps_ln_out(ps2_z)
        ps2_logits = self.ps_head(ps2_z) # prediction logits from speech

        return ps_logits, ps2_logits, vq_loss, encodings
    
    @torch.no_grad()
    def predict_from_speech(self, sentences_idx):

        """
        IN prediction from speech alone (speech -> reconstructed boards)

        Args:
            sentences_idx (torch.Tensor): input tensor of shape (batch_size, seq_length), represents speech
        Returns:
            ps2_logits (torch.Tensor): logits produced by IN from speech alone (output of ALN)
        """
        
        device = sentences_idx.device
        batch_size, seq_length = sentences_idx.size()

        # make picutre and speech position embeddings
        position_embeddings_picture = torch.arange(0, self.config.block_size, dtype=torch.long, device=device).unsqueeze(0) # idx of position, shape (1, t)
        position_embeddings_picture = self.ps_pos_emb(position_embeddings_picture) # shape (1, t, d)

        position_embeddings_speech = torch.arange(0, self.config.block_speech_size, dtype=torch.long, device=device).unsqueeze(0) # idx of position, shape (1, t)
        position_embeddings_speech = self.ss_pos_emb(position_embeddings_speech) # shape (1, t, d)


        ss_vq = self.ss_vector_quantizer.get_codebook_entries(sentences_idx)
        ss_vq = ss_vq.to(device)

        # initialize picture
        # get token embeddings
        ps2_z_idx = (self.config.vocab_size - 1) * torch.ones((batch_size, self.config.block_size)).type(torch.LongTensor).to(device)
        ps2_z = self.ps_token_embedding(ps2_z_idx)  # each index maps to a (learnable) vector

        # add position embeddings
        ps2_z = ps2_z + position_embeddings_picture # broadcasting (B,block_size,n_embd)
        ps2_z = self.ps_drop(ps2_z)

        # initialize speech
        ps2_q = ss_vq + position_embeddings_speech # broadcasting
        
        for block in self.ps_blocks:
            ps2_z = block(ps2_z, ps2_q) # /!\ cross attention on different block sizes

        ps2_z = self.ps_ln_out(ps2_z)
        ps2_logits = self.ps_head(ps2_z) # prediction logits from speech

        return ps2_logits

class LEAFactor(nn.Module):

    """
    Implementation of the language-enhanced transformer architecture (LEA) with factorized token/positional embeddings.
    """

    def __init__(self, vocab_size, block_size, n_layer=2, n_head=4, n_embd=64, n_embd_pos = 32,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0, vocab_speech_size=50, block_speech_size = 10, vq_type = 'EMA', bool_blank_speech = True):
        
        """
        Creates a LEAFactor object.
        
        Args:
            vocab_size (int): size of the token vocabulary
            block_size (int): size of the token sequence
            n_layer (int): number of transformer blocks
            n_head (int): number of attention heads
            n_embd (int): embedding dimension
            embd_pdrop (float): dropout probability for the embedding layer
            resid_pdrop (float): dropout probability for the residual layer
            attn_pdrop (float): dropout probability for the attention layer
            n_unmasked (int): number of tokens to leave unmasked in the causal mask (= block_size), kept for legacy reasons
            vocab_speech_size (int): size of the speech token vocabulary
            block_speech_size (int): length of a sentence
            vq_type (str): type of vector quantization to use (vanilla or EMA)
            bool_blank_speech (bool): whether to use blank speech or not

        Returns:
            None
        """
        
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, # board token vocab
                           block_size=block_size,
                           n_layer=n_layer, 
                           n_head=n_head, 
                           n_embd=n_embd,
                           n_embd_pos=n_embd_pos,
                           embd_pdrop=embd_pdrop, 
                           resid_pdrop=resid_pdrop, 
                           attn_pdrop=attn_pdrop,
                           n_unmasked=n_unmasked, # depreciated
                           vocab_speech_size=vocab_speech_size, # speech token vocab
                           block_speech_size=block_speech_size,
                           vq_type = vq_type,
                           bool_blank_speech = bool_blank_speech)
        
        #........................................
        # picture stream "ps" (masked board / speech -> reconstructed board), Inference Network (IN)
        #........................................
        
        # embeddings
        self.ps_token_embedding = nn.Embedding(config.vocab_size, config.n_embd) # token embeddings
        self.ps_pos_emb = nn.Embedding(config.block_size, config.n_embd_pos) # position encodings
        self.ps_drop = nn.Dropout(config.embd_pdrop)

        # transformer
        # using decoder because it should cross attend to speech
        self.ps_blocks = nn.ModuleList([BlockDecoderFactor(config) for _ in range(config.n_layer)])

        # head
        # convert representations in logits over vocabulary for each picture token
        self.ps_ln_out = nn.LayerNorm(config.n_embd)
        self.ps_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        #........................................
        # speech stream (IN latent representations -> speech), Auxiliary Language Network (ALN)
        #........................................

        # embeddings
        if vq_type == 'EMA':
            print('using VectorQuantizerEMA')
            self.ss_vector_quantizer = VectorQuantizerEMA(config.vocab_speech_size, config.n_embd, commitment_cost = 0.25, usage_cost=0.25)
        elif vq_type == 'vanilla':
            print('using VectorQuantizer')
            self.ss_vector_quantizer = VectorQuantizer(config.vocab_speech_size, config.n_embd, commitment_cost = 0.25, usage_cost=0.25)
        else:
            raise ValueError('vq_type not recognized')
        
        # what will be used as the initial token when generating speech
        self.ss_token_init = nn.Embedding(config.block_speech_size, config.n_embd) # will be learnt
        self.ss_pos_emb = nn.Embedding(config.block_speech_size, config.n_embd_pos)  
        self.ss_drop = nn.Dropout(config.embd_pdrop)

        # transformer
        # using decoder because it should cross attend to picture stream
        self.ss_blocks = nn.ModuleList([BlockDecoderFactor(config) for _ in range(config.n_layer)])

        # head
        # convert representations in logits over vocabulary for each picture token
        self.ss_ln_out = nn.LayerNorm(config.n_embd)

        #........................................
        # Initialize weights
        #........................................

        self.apply(self._init_weights) # init weights recursively
        self.config = config

    def _init_weights(self, module):

        """
        Initialize the weights of the model.
        """

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_idx):

        """
        Forward pass of the LEA model.

        Args:
            input_idx (torch.Tensor): input tensor of shape (batch_size, seq_length), repesents boards
        
        Returns:
            ps_logits (torch.Tensor): logits produced by IN from masked boards
            ps2_logits (torch.Tensor): logits produced by IN from speech alone (output of ALN)
            vq_loss (torch.Tensor): loss of the vector quantizer
            encodings (torch.Tensor): one-hot encodings of the speech tokens
        """
        
        device = input_idx.device
        batch_size, seq_length = input_idx.size()

        # make picutre and speech position embeddings
        position_embeddings_picture = torch.arange(0, self.config.block_size, dtype=torch.long, device=device).unsqueeze(0) # idx of position, shape (1, t)
        position_embeddings_picture = self.ps_pos_emb(position_embeddings_picture) # shape (1, t, d)

        position_embeddings_speech = torch.arange(0, self.config.block_speech_size, dtype=torch.long, device=device).unsqueeze(0) # idx of position, shape (1, t)
        position_embeddings_speech = self.ss_pos_emb(position_embeddings_speech) # shape (1, t, d)

        #-------------------------------------
        # IN: Reconstruct board from masked input
        # with blank speech (masked boards -> reconstructed boards)
        #-------------------------------------

        # initialize picture
        ps_z = self.ps_token_embedding(input_idx)  # each index maps to a (learnable) vector
        ps_z = self.ps_drop(ps_z)

        # initialize speech
        if self.config.bool_blank_speech:
            ps_q = torch.zeros((batch_size, self.config.block_speech_size, self.config.n_embd)).to(device)
        else:# could also use the last token in vocabulary
            ps_q_idx = (self.config.vocab_speech_size - 1) * torch.ones((batch_size, self.config.block_speech_size)).type(torch.LongTensor).to(device)
            ps_q = self.ss_vector_quantizer.get_codebook_entries(ps_q_idx)
        
        for block in self.ps_blocks:
            ps_z = block(ps_z, ps_q, position_embeddings_picture, position_embeddings_speech) # /!\ cross attention on different block sizes

        ps_z = self.ps_ln_out(ps_z)
        ps_logits = self.ps_head(ps_z) # prediction logits from picture

        #-------------------------------------
        # ALN: Construct speech from internal representations 
        # of the picture stream (IN latent representations -> speech)
        #-------------------------------------

        # initialize speech
        ss_z_idx = torch.arange(0, self.config.block_speech_size, dtype=torch.long, device=device).unsqueeze(0)
        ss_z = self.ss_token_init(ss_z_idx) # (1, block_speech_size, n_embd_speech)
        ss_z = repeat(ss_z, '1 n d -> b n d', b = batch_size)
        ss_z = self.ss_drop(ss_z)

        for block in self.ss_blocks:
            ss_z = block(ss_z, ps_z, position_embeddings_speech, position_embeddings_picture) # /!\ cross attention on different block sizes

        ss_z = self.ss_ln_out(ss_z)

        # vector quatize to get discrete tokens
        ss_vq, vq_loss, encodings, encoding_indices = self.ss_vector_quantizer(ss_z)
        
        #-------------------------------------
        # IN: Reconstruct board from speech alone
        # (speech -> reconstructed boards)
        #------------------------------------- 

        # initialize picture
        # get token embeddings
        ps2_z_idx = (self.config.vocab_size - 1) * torch.ones((batch_size, seq_length)).type(torch.LongTensor).to(device)
        ps2_z = self.ps_token_embedding(ps2_z_idx)  # each index maps to a (learnable) vector
        ps2_z = self.ps_drop(ps2_z)

        # initialize speech
        ps2_q = ss_vq
        
        for block in self.ps_blocks:
            ps2_z = block(ps2_z, ps2_q, position_embeddings_picture, position_embeddings_speech) # /!\ cross attention on different block sizes

        ps2_z = self.ps_ln_out(ps2_z)
        ps2_logits = self.ps_head(ps2_z) # prediction logits from speech

        return ps_logits, ps2_logits, vq_loss, encodings
    
    @torch.no_grad()
    def predict_from_board(self, input_idx):

        """
        IN prediction from masked board (masked board -> reconstructed boards)

        Args:
            input_idx (torch.Tensor): input tensor of shape (batch_size, seq_length), represents boards
        Returns:
            ps_logits (torch.Tensor): logits produced by IN from masked boards
        """

        device = input_idx.device
        batch_size, seq_length = input_idx.size()

        # make picutre and speech position embeddings
        position_embeddings_picture = torch.arange(0, self.config.block_size, dtype=torch.long, device=device).unsqueeze(0) # idx of position, shape (1, t)
        position_embeddings_picture = self.ps_pos_emb(position_embeddings_picture) # shape (1, t, d)

        position_embeddings_speech = torch.arange(0, self.config.block_speech_size, dtype=torch.long, device=device).unsqueeze(0) # idx of position, shape (1, t)
        position_embeddings_speech = self.ss_pos_emb(position_embeddings_speech) # shape (1, t, d)

        #-------------------------------------
        # Reconstruct board from masked input
        # with blank speech
        #-------------------------------------

        # INIT PICTURE STREAM
        # print(f'ckpt 1: creating ps_z\n... batch_size = {batch_size}, seq_length = {seq_length}')
        ps_z = self.ps_token_embedding(input_idx)  # each index maps to a (learnable) vector
        # print(f'... ps_token_embeddings = {ps_token_embeddings.size()}')
        ps_z = self.ps_drop(ps_z)

        # INIT BLANK SPEECH
        if self.config.bool_blank_speech:
            ps_q = torch.zeros((batch_size, self.config.block_speech_size, self.config.n_embd)).to(device)
        else:# could also use the last token in vocabulary
            ps_q_idx = (self.config.vocab_speech_size - 1) * torch.ones((batch_size, self.config.block_speech_size)).type(torch.LongTensor).to(device)
            ps_q = self.ss_vector_quantizer.get_codebook_entries(ps_q_idx)
        
        # print('\n*** ps_blocks start ***')
        for block in self.ps_blocks:
            ps_z = block(ps_z, ps_q, position_embeddings_picture, position_embeddings_speech) # /!\ cross attention on different block sizes

        ps_z = self.ps_ln_out(ps_z)
        ps_logits = self.ps_head(ps_z) # prediction logits from picture
        
        return ps_logits

    @torch.no_grad()
    def predict_from_speech(self, sentences_idx):

        """
        IN prediction from speech alone (speech -> reconstructed boards)

        Args:
            sentences_idx (torch.Tensor): input tensor of shape (batch_size, seq_length), represents speech
        Returns:
            ps2_logits (torch.Tensor): logits produced by IN from speech alone (output of ALN)
        """
        
        device = sentences_idx.device
        batch_size, _ = sentences_idx.size()

        # make picutre and speech position embeddings
        position_embeddings_picture = torch.arange(0, self.config.block_size, dtype=torch.long, device=device).unsqueeze(0) # idx of position, shape (1, t)
        position_embeddings_picture = self.ps_pos_emb(position_embeddings_picture) # shape (1, t, d)

        position_embeddings_speech = torch.arange(0, self.config.block_speech_size, dtype=torch.long, device=device).unsqueeze(0) # idx of position, shape (1, t)
        position_embeddings_speech = self.ss_pos_emb(position_embeddings_speech) # shape (1, t, d)


        ss_vq = self.ss_vector_quantizer.get_codebook_entries(sentences_idx)
        ss_vq = ss_vq.to(device)

        # initialize picture
        # get token embeddings
        ps2_z_idx = (self.config.vocab_size - 1) * torch.ones((batch_size, self.config.block_size)).type(torch.LongTensor).to(device)
        ps2_z = self.ps_token_embedding(ps2_z_idx)  # each index maps to a (learnable) vector
        ps2_z = self.ps_drop(ps2_z)

        # initialize speech
        ps2_q = ss_vq
        
        for block in self.ps_blocks:
            ps2_z = block(ps2_z, ps2_q, position_embeddings_picture, position_embeddings_speech) # /!\ cross attention on different block sizes

        ps2_z = self.ps_ln_out(ps2_z)
        ps2_logits = self.ps_head(ps2_z) # prediction logits from speech

        return ps2_logits

#/////////////////////////////////////////////////////////////////////////
# main
#/////////////////////////////////////////////////////////////////////////

if __name__ == "__main__":

    #-------------------------------------
    # Test forward pass
    #-------------------------------------

    batch, seq_length, vocab_size = 2, 64, 25

    speaker_model = LEAFactor(vocab_size = vocab_size, 
                                block_size = seq_length, 
                                n_layer=3, 
                                n_head=2, 
                                n_embd=16,
                                n_embd_pos=8, 
                                embd_pdrop=0., 
                                resid_pdrop=0., 
                                attn_pdrop=0., 
                                n_unmasked=0, 
                                vocab_speech_size=50, 
                                block_speech_size=4)

    device = 'cpu'
    speaker_model = speaker_model.to(device)
    input_idx = torch.randint(0, vocab_size, (batch, seq_length)).to(device)

    ps_logits, ps2_logits, vq_loss, encodings = speaker_model(input_idx)
    print(f'ps_logits:{ps_logits.size()}')
    print(f'ps2_logits:{ps2_logits.size()}')
    print(f'vq_loss:{vq_loss}')
    print(f'encodings:{encodings.size()}')