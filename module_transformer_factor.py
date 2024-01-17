"""
Implementation of the transformer class with factorized token and position embeddings (GPTFactor).
Code adapted from minGPT (https://github.com/karpathy/minGPT).
"""

#----------------------------------------------------
# import libraries
#----------------------------------------------------
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
        Returns:
            None
        """

        self.vocab_size = vocab_size
        self.block_size = block_size

        for k, v in kwargs.items():
            setattr(self, k, v)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head self-attention block with linear projection at the end.
    Note: I have removed causal mask from this implementation.
    """

    def __init__(self, config):

        """
        Create a CausalSelfAttention object.

        Args:
            config (GPTConfig): configuration of the model
        Returns:
            None
        """

        super().__init__()
        assert config.n_embd % config.n_head == 0 # n_embd must be divisible by n_head

        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd + config.n_embd_pos, config.n_embd)
        self.query = nn.Linear(config.n_embd + config.n_embd_pos, config.n_embd)
        self.value = nn.Linear(config.n_embd + config.n_embd_pos, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # We set all entries of mask to 1 => no causal mask
        # We keep this for legacy reasons and future developments
        mask = torch.tril(torch.ones(config.block_size,
                                     config.block_size)) # square matrix with 1 on and under the diagonal 
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, xp):

        """
        Forward pass.

        Args:
            xp (tensor): input tensor of shape (batch_size, block_size, n_embd + n_embd_pos)
        Returns:
            y (tensor): output tensor of shape (batch_size, block_size, n_embd)
        """

        B, T, _ = xp.size() # batch and block sizes
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(xp).view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(xp).view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(xp).view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf')) # nothing happens when config.n_unmasked == T (block size)

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))

        return y

class Block(nn.Module):
    """
    Transformer block consisting of a causal self-attention layer and a mlp that feed on a residual stream. 
    """

    def __init__(self, config):

        """
        Create a Block object.

        Args:
            config (GPTConfig): configuration of the model
        Returns:
            None
        """

        super().__init__()

        # layer norms
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        # attn and MLP blocks
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd + config.n_embd_pos, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, p, activations = None, verbose = False):
        """ 
        Forward pass.

        Args:
            x (tensor): input tensor of shape (batch_size, block_size, n_embd)
            p (tensor): position encoding tensor of shape (1, block_size, n_embd_pos)
            activations (dict): dictionary of activations to edit the activations of the model (default: None)
            verbose (bool): whether to print the edited activations (default: False)
        Returns:
            x (tensor): output tensor of shape (batch_size, block_size, n_embd)
        """
        
        # broadcast along batch dimension to match batch size of x
        p_batch = repeat(p, '1 l d -> b l d', b = x.size(0))

        if activations is not None: # forward pass with edited activations

            # z
            z = self.ln1(x)
            if 'z' in activations:
                if verbose: print(f'... editing z')
                z = activations['z']
            xp = torch.cat((z, p_batch), dim = 2)

            # attn_update
            attn_update =  self.attn(xp)
            if 'attn_update' in activations:
                if verbose: print(f'... editing attn_update')
                attn_update = activations['attn_update']
            x = x + attn_update

            # mlp_in
            mlp_in = self.ln2(x)
            if 'mlp_in' in activations:
                if verbose: print(f'... editing mlp_in')
                mlp_in = activations['mlp_in']  
            xp = torch.cat((mlp_in, p_batch), dim = 2)

            # mlp_update
            mlp_update = self.mlp(xp)
            if 'mlp_update' in activations:
                if verbose: print(f'... editing mlp_update')
                mlp_update = activations['mlp_update']
            x = x + mlp_update

            # z_attn_mlp
            if 'z_attn_mlp' in activations:
                if verbose: print(f'... editing z_attn_mlp')
                x = activations['z_attn_mlp']
        
        else: # original forward pass

            xp = torch.cat((self.ln1(x), p_batch), dim = 2)
            x = x + self.attn(xp)
            xp = torch.cat((self.ln2(x), p_batch), dim = 2)
            x = x + self.mlp(xp)

        return x

class GPTFactor(nn.Module):

    """
    GPT model with factorized token and position embeddings.
    """
    
    def __init__(self, vocab_size, block_size, 
                 n_layer=2, n_head=4, n_embd=64, n_embd_pos = 32,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0):
        """
        Create a GPTFactor object.

        Args:
            vocab_size (int): size of the token vocabulary
            block_size (int): size of the token sequence (flattened the board of tokens)
            n_layer (int): number of transformer blocks
            n_head (int): number of attention heads
            n_embd (int): embedding dimension
            n_embd_pos (int): embedding dimension for the position encodings
            embd_pdrop (float): dropout probability for the embedding layer
            resid_pdrop (float): dropout probability for the residual layer
            attn_pdrop (float): dropout probability for the attention layer
            n_unmasked (int): number of tokens to leave unmasked in the causal mask (= block_size), kept for legacy reasons
        Returns:
            None
        """
        super().__init__()
        # configuration
        config = GPTConfig(vocab_size=vocab_size, 
                           block_size=block_size,
                           embd_pdrop=embd_pdrop, 
                           resid_pdrop=resid_pdrop, 
                           attn_pdrop=attn_pdrop,
                           n_layer=n_layer, 
                           n_head=n_head, 
                           n_embd=n_embd,
                           n_embd_pos=n_embd_pos, 
                           n_unmasked=n_unmasked)
        
        self.block_size = config.block_size
        self.config = config

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd_pos))  
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # initialize weights
        self.apply(self._init_weights) 

    def get_block_size(self):
        """
        Returns the block size.
        """
        return self.block_size

    def _init_weights(self, module):
        """
        Initializes the weights.

        Args:
            module (nn.Module): module to initialize
        Returns:
            None
        """

        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None, activations = None, verbose = False):

        """
        Forward pass.

        Args:
            idx (tensor): input tensor of tokens, shape (batch_size, block_size)
            embeddings (tensor): explicit embeddings to prepend to the input tensor (default: None)
            activations (dict): dictionary of activations to edit the activations of the model (default: None)
            verbose (bool): whether to notify edits (default: False)
        Returns:
            logits (tensor): output tensor of shape (batch_size, block_size, vocab_size)
        """
        # get token embeddings
        token_embeddings = self.tok_emb(idx) 

        # prepend explicit embeddings
        if embeddings is not None: 
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        x = self.drop(token_embeddings)

        if activations is not None: # forward pass with edited activations

            for idx_block, block in enumerate(self.blocks):

                if f'b{idx_block}' in activations:
                    if verbose: print(f'* editing Block {idx_block}')
                    x = block(x, self.pos_emb, activations = activations[f'b{idx_block}'], verbose = verbose)
                else:
                    x = block(x, self.pos_emb)

            x = self.ln_f(x)
            if 'z_out' in activations:
                if verbose: print(f'* editing z_out')
                x = activations['z_out']
            
        else: # original forward pass
            for block in self.blocks:
                x = block(x, self.pos_emb)
            x = self.ln_f(x)
            
        logits = self.head(x)
        return logits
#----------------------------------------------------

#----------------------------------------------------
# test code integrity
#----------------------------------------------------
if __name__ == '__main__':
    # test the forward pass
    transformer = GPTFactor(vocab_size = 20, block_size = 10, n_layer = 2, n_head = 2, n_embd = 32, n_embd_pos = 16)
    idx = torch.randint(0, 20, (2, 10))
    logits = transformer(idx)
    print(logits.shape)
#----------------------------------------------------       