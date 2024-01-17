"""
Implements the `TrainTransformer` class, a wraper to train transformers and probe their functions.
"""

#----------------------------------------------------
# import libraries
#----------------------------------------------------
import time
import wandb
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from einops import rearrange, reduce, repeat
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from module_transformer_factor import GPTFactor
from utils import find_best_plot_dim, get_ij, merge_attn_plots, divide_into_batches
#----------------------------------------------------

#----------------------------------------------------
# helper functions
#----------------------------------------------------
def torch_entropy(logits):

    """
    computes the entropy of a discrete probability distribution defined by softmax(logits).

    Args:
        logits: tensor of logits (batch, sequence lenght, value)
    Returns:
        entropy: tensor of normalized entropy (batch, sequence lenght)
    """
    
    distribution_size = logits.size(-1)

    # compute reference entropy for a uniform distribution with `distribution_size`
    ref_proba = F.softmax(torch.ones((1,distribution_size)), dim = -1) + 1e-10
    ref_entropy = (-torch.sum(ref_proba * torch.log2(ref_proba), dim = -1)).item()
    
    # get probability distribution from logits
    proba = F.softmax(logits, dim = -1) + 1e-10

    # compute normalized entropy
    return -torch.sum(proba * torch.log2(proba), dim = -1) / ref_entropy

def dist(X):

    """
    Computes the distance matrix between all rows of X
    
    Args:
        X: numpy array of shape (n,d)
    Returns:
        D: numpy array of shape (n,n), distance matrix (upper triangular)
    """
    
    n, d = X.shape
    D = np.zeros((n,n))
    
    for i in range(n):
        for j in range(i,n):
            D[i,j] = np.sqrt(np.sum((X[i]-X[j])**2))
            
    return D

def dist_unit(X):

    """
    Computes the distance matrix between all rows of X, after converting each row to a unit vector

    Args:
        X: numpy array of shape (n,d)
    Returns:
        D: numpy array of shape (n,n), distance matrix (upper triangular)
    """
    
    n, d = X.shape
    D = np.zeros((n,n))

    # convert in unit vectors
    X_unit = np.copy(X)
    for i in range(n):
        X_unit[i] = X_unit[i] / np.linalg.norm(X_unit[i])
    
    # compute distances in upper diagonal
    for i in range(n):
        for j in range(i,n):
            D[i,j] = np.sqrt(np.sum((X_unit[i]-X_unit[j])**2))
            
    return D

def dist_cosine(X):

    """
    Computes the cosine similarity matrix between all rows of X.

    Args:
        X: numpy array of shape (n,d)
    Returns:
        D: numpy array of shape (n,n), distance matrix (upper triangular)
    """

    n, d = X.shape
    D = np.zeros((n,n))

    X_unit = np.copy(X)
    for i in range(n):
        X_unit[i] = X_unit[i] / np.linalg.norm(X_unit[i])
    
    for i in range(n):
        for j in range(i,n):
            D[i,j] = np.dot(X_unit[i], X_unit[j])
            
    return D

def ramp(x, val_s, val_e, step_s, step_e):

    """
    Implements a ramp function between val_s and val_e for x between step_s and step_e
    
    Args:
        x (float): current step
        val_s (float): value at step_s, start of ramp
        val_e (float): value at step_e, end of ramp
        step_s (float): start of ramp
        step_e (float): end of ramp
    Returns:
        val (float): value of the ramp at x
    """

    if x < step_s:
        return val_s
    elif step_s <= x  and x <= step_e:
        return val_s + (val_e - val_s) * (x - step_s) / (step_e-step_s)
    else:
        return val_e
#----------------------------------------------------

#----------------------------------------------------
# class definition
#----------------------------------------------------
class TrainTransformer:

    """wraper to train and probe transformer"""
    
    def __init__(self, dataset, **kwargs):

        """
        Create a `TrainTransformer` object.

        Args:
            dataset: a `Dataset` object
            **kwargs: keyword arguments to be passed to the `GPTFactor` constructor
                - 'transformer_type' (str): type of transformer, 'factor' is the only one supported at the moment
                - 'vocab_size' (int): size of entire vocab (background + object + UNK)
                - 'block_size' (int): size of a context (board_dim**2)
                - 'n_layer' (int): number of transformer layers
                - 'n_head' (int): number of heads in multi-head attention
                - 'n_embd' (int): number embedding dimensions
                - 'n_embd_pos' (int): number of positional encoding dimensions
                - 'embd_pdrop' (floar): dropout rate for embeddings
                - 'resid_pdrop' (floar): dropout rate for residual connections
                - 'attn_pdrop' (floar): dropout rate for attention
                - 'n_unmasked'(int): number of unmasked token for causal attention, depreciated & overwritten
                - 'masking_token' (int): index of UNK. token
                - 'masking_p' (floar): final token masking probability
                - 'masking_patch_dim' (int): height and width for patch masking, no longer used
                - 'masking_patch_epoch' (int): epoch at which masking reach full proba & patch masking starts
                - 'masking_patch_p' (floar): probability of patch masking once patch masking starts
                - 'board_curriculum' (bool): whether to use curriculum learning
                - 'board_proba_epoch_start' (int): epoch at which board curriculum starts
                - 'board_proba_epoch_end' (int): epoch at which board curriculum ends
                - 'board_proba_start' (list of float): probabilities of generating each board type (single, double, composition) at curriculum start
                - 'board_proba_end' (list of float): probabilities of generating each board type (single, double, composition) at curriculum end
                - 'learning_rate' (float): learning rate
                - 'batch_size' (int): batch size
                - 'epochs' (int): total number of epochs
                - 'device' (float): device used for training
                - 'plot_dpi' (int): dpi for plots                
                - 'epoch2print' (int): loss print frequency
                - 'epoch2test' (int): test loss print frequency
                - 'save_model' (bool): whether to save model or not
                - 'epoch2save' (int): model save frequency
                - 'run_name' (str): name of the run
        Returns:
            None
        """

        # save all arguments as attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.dataset = dataset

        # instantiate transformer
        if self.transformer_type == 'factor':
            print('Using GPTFactor')
            self.transformer = GPTFactor(vocab_size = self.vocab_size, 
                               block_size = self.block_size, 
                               n_layer = self.n_layer, 
                               n_head = self.n_head, 
                               n_embd = self.n_embd,
                               n_embd_pos = self.n_embd_pos,
                               embd_pdrop = self.embd_pdrop, 
                               resid_pdrop = self.resid_pdrop, 
                               attn_pdrop = self.attn_pdrop, 
                               n_unmasked = self.n_unmasked).double().to(self.device)
        else:
            raise ValueError('transformer_type must be "factor", "vanilla" is depreciated')
        
        # setup optimizer
        self.optim = self._configure_optimizers()
        # add other attributes useful for training
        self.step = 0
        self.epoch = 0

        # create hooks to recover activations and corresponding dictionaries that will store them
        self._register_hooks_attn()
        self._register_hooks_z()
        self._register_hooks_svg()
        self._register_hooks_pca()
        self._unregister_hooks()
        
    def load_transformer(self, filepath):
        """
        Load a pretrained transformer

        Args:
            filepath (str): path to the transformer checkpoint
        Returns:
            None
        """
        self.transformer.load_state_dict(torch.load(filepath, map_location=self.device))
    
    #.......................................
    # register hooks to recover activations
    # Each function registers a specific set of hooks to recover specific activations along the forward pass. These functions were created for various analyses. 
    #.......................................
    
    def _register_hooks_attn(self):
        
        """Adds hooks to recover activations of the attention matrices"""
        
        self.activations_attn = {} # will store the activations during the forward pass
        def getActivation(name):
            def hook(model, input, output):
                self.activations_attn[name] = output.detach()
            return hook
        
        self.list_hooks_attn = [] # stores the hooks to be removed later
        for i in range(self.n_layer):
            self.list_hooks_attn.append(self.transformer.blocks[i].attn.attn_drop.register_forward_hook(getActivation(f'attn_{i}')))

    def _register_hooks_z(self):
        
        """Adds hooks to recover activations throughout the transformer"""

        self.activations_z = {}
        def getActivation(name):
            def hook(model, input, output):
                self.activations_z[name] = output.detach()
            return hook
        
        self.list_hooks_z = []
        for i in range(self.n_layer):
            self.list_hooks_z.append(self.transformer.blocks[i].ln1.register_forward_hook(getActivation(f'b_{i}_z')))
            self.list_hooks_z.append(self.transformer.blocks[i].attn.register_forward_hook(getActivation(f'b_{i}_attn')))
            self.list_hooks_z.append(self.transformer.blocks[i].ln2.register_forward_hook(getActivation(f'b_{i}_z_attn')))
            self.list_hooks_z.append(self.transformer.blocks[i].mlp.register_forward_hook(getActivation(f'b_{i}_mlp')))
            self.list_hooks_z.append(self.transformer.blocks[i].register_forward_hook(getActivation(f'b_{i}_z_attn_mlp')))

    def _register_hooks_svg(self):
        
        """Add hooks to recover activations throughout the transformer, to make svg plots"""

        self.activations_svg = {}
        def getActivation(name):
            def hook(model, input, output):
                self.activations_svg[name] = output.detach()
            return hook
        
        def getInput(name):
            def hook(model, input, output):
                self.activations_svg[name] = input[0].detach()
            return hook
        
        self.list_hooks_svg = []
        for i in range(self.n_layer):

            self.list_hooks_svg.append(self.transformer.blocks[i].ln1.register_forward_hook(getActivation(f'svg_b{i}_z')))
            self.list_hooks_svg.append(self.transformer.blocks[i].attn.register_forward_hook(getActivation(f'svg_b{i}_attn_update')))

            self.list_hooks_svg.append(self.transformer.blocks[i].attn.attn_drop.register_forward_hook(getActivation(f'svg_b{i}_a')))
            self.list_hooks_svg.append(self.transformer.blocks[i].attn.value.register_forward_hook(getActivation(f'svg_b{i}_v')))
            self.list_hooks_svg.append(self.transformer.blocks[i].attn.proj.register_forward_hook(getInput(f'svg_b{i}_attn_v')))
            self.list_hooks_svg.append(self.transformer.blocks[i].ln2.register_forward_hook(getInput(f'svg_b{i}_z_attn')))

            self.list_hooks_svg.append(self.transformer.blocks[i].ln2.register_forward_hook(getActivation(f'svg_b{i}_mlp_in')))
            self.list_hooks_svg.append(self.transformer.blocks[i].mlp.register_forward_hook(getActivation(f'svg_b{i}_mlp_update')))
            self.list_hooks_svg.append(self.transformer.blocks[i].register_forward_hook(getActivation(f'svg_b{i}_z_attn_mlp')))
    
    def _register_hooks_pca(self):
        
        """add hooks to recover activations throughout the transformer, for pca analysis"""

        self.activations_pca = {}
        def getActivation(name):
            def hook(model, input, output):
                self.activations_pca[name] = output.detach()
            return hook
        
        def getInput(name):
            def hook(model, input, output):
                self.activations_pca[name] = input[0].detach()
            return hook
        
        self.list_hooks_pca = []
        for i in range(self.n_layer):

            self.list_hooks_pca.append(self.transformer.blocks[i].ln1.register_forward_hook(getActivation(f'b{i}_z')))
            self.list_hooks_pca.append(self.transformer.blocks[i].attn.register_forward_hook(getActivation(f'b{i}_attn_update')))

            self.list_hooks_pca.append(self.transformer.blocks[i].attn.attn_drop.register_forward_hook(getActivation(f'b{i}_a')))
            self.list_hooks_pca.append(self.transformer.blocks[i].attn.value.register_forward_hook(getActivation(f'b{i}_v')))
            self.list_hooks_pca.append(self.transformer.blocks[i].attn.proj.register_forward_hook(getInput(f'b{i}_attn_v')))
            self.list_hooks_pca.append(self.transformer.blocks[i].ln2.register_forward_hook(getInput(f'b{i}_z_attn')))

            self.list_hooks_pca.append(self.transformer.blocks[i].ln2.register_forward_hook(getActivation(f'b{i}_mlp_in')))
            self.list_hooks_pca.append(self.transformer.blocks[i].mlp.register_forward_hook(getActivation(f'b{i}_mlp_update')))
            self.list_hooks_pca.append(self.transformer.blocks[i].register_forward_hook(getActivation(f'b{i}_z_attn_mlp')))
        
        # add hook on the output of the ln_f
        self.list_hooks_pca.append(self.transformer.ln_f.register_forward_hook(getActivation(f'z_out')))
        self.list_hooks_pca.append(self.transformer.head.register_forward_hook(getActivation(f'logit')))

    def _unregister_hooks(self):

        """Removes all hooks from the model"""

        if hasattr(self, "list_hooks"):
            for h in self.list_hooks_attn:
                h.remove()
            self.list_hooks_attn = []
        if hasattr(self, "list_hooks_z"):
            for h in self.list_hooks_z:
                h.remove()
            self.list_hooks_z = []
        if hasattr(self, "list_hooks_svg"):
            for h in self.list_hooks_svg:
                h.remove()
            self.list_hooks_svg = []
        if hasattr(self, "list_hooks_pca"):
            for h in self.list_hooks_pca:
                h.remove()
            self.list_hooks_pca = []
    
    #.......................................
    # optimizer
    #.......................................
    
    def _configure_optimizers(self):

        """Configures the optimizer"""

        decay, no_decay = set(), set() # will contain the name of parameters to be decayed or not
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")

        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        
        # added from minGPT on 09/28/2023
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        #.......................................

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer
    
    #.......................................
    # board masking
    #.......................................
    
    def _mask_input(self, board, masking_p, patch_masking = False, patch_masking_p = 0.1, bool_report = False):

        """
        Masks tokens of a board with masking_p probability. If `patch_masking` is True, larger patches of the board are masked with a probability of `patch_masking_p`.

        Args:
            board: tensor of shape (batch, board_dim, board_dim), boards to be masked
            masking_p: float, probability of masking a token
            patch_masking: bool, whether or not to do patch masking
            patch_masking_p: float, probability of patch masking
            bool_report: bool, whether or not to return stats about the masking
        Returns:
            masked_board: tensor of shape (batch, board_dim, board_dim), masked boards
            stats: dict, stats about the masking (only if bool_report is True)

        Note:
            - /!\ Patch masking for boards with composite abstraction only work for 2x2 composites as used in the paper.
        """
        
        # generate masked version of the board by sustituting the UNK. token
        fully_masked_board = self.masking_token * torch.ones(board.shape, device = self.device)
        mask = torch.bernoulli(masking_p * torch.ones(board.shape, device = self.device))
        masked_board = (mask * fully_masked_board + (1-mask) * board).long() # should be indices

        display = False # for debugging
        # decide whether or not to do patch masking
        # make a decision for each board in the batch
        do_patch_masking = torch.bernoulli(patch_masking_p * torch.ones((board.shape[0],)))

        # stats
        stats_boardType = np.zeros((board.shape[0],))
        
        if patch_masking: # perfom patch masking in addition to single token masking

            for i in range(board.shape[0]): # iterate over batch

                if do_patch_masking[i] == 1: # do patch masking

                    # board will be treated differently whether it features a composite abstraction or not. For boards with single/ double root abstractions, patch is chosen randomly. For boards with composite abstractions, patch is chosen among the 4 possible positions to conver one of the constituent abstractions. Patch will have the same dimension as a root abstraction.

                    # get abstraction mask
                    abs_mask = board[i] >= self.dataset.vocab_bck
                    
                    if torch.sum(abs_mask) > 2 * self.dataset.abs_dim**2: # composite abstraction

                        #print(f'found a composite abstraction, board {i}:\n{board[i]}')
                        stats_boardType[i] = 3

                        # get the coordinate of True values in abs_mask
                        abs_coord = torch.nonzero(abs_mask, as_tuple=False)

                        # get min and max coordinates on each axis
                        min_coord = torch.min(abs_coord, dim=0)[0]
                        max_coord = torch.max(abs_coord, dim=0)[0]
                        #print(f'min coord: {min_coord}, max coord: {max_coord}')

                        #/!\ only work for specific 2x2 composites 
                        patch_positions = torch.tensor([
                            [min_coord[0], min_coord[1]],
                            [min_coord[0], max_coord[1] - self.dataset.abs_dim + 1],
                            [max_coord[0] - self.dataset.abs_dim + 1, min_coord[1]],
                            [max_coord[0] - self.dataset.abs_dim + 1, max_coord[1] - self.dataset.abs_dim + 1]
                        ])
                        #print(f'patch positions: {patch_positions}')

                        # randomly choose a patch position
                        patch_pos = patch_positions[torch.randint(patch_positions.size(0), (1,)).item()]
                        i_s, i_e = patch_pos[0], patch_pos[0] + self.dataset.abs_dim
                        j_s, j_e = patch_pos[1], patch_pos[1] + self.dataset.abs_dim
                        #print(f'coordinates: {i_s, i_e, j_s, j_e}')
                        masked_board[i,i_s:i_e,j_s:j_e] = self.masking_token

                        if display:
                            print(f'patching board {i}')
                            print(f'{board[i]}')
                            print(f'{masked_board[i]}')
                            display = False

                    else: # single or multiple root abstractions

                        stats_boardType[i] = 1

                        # choose a position where to mask
                        patch_pos_i = torch.randint(board.shape[1] - self.dataset.abs_dim + 1, (1,)).item()
                        patch_pos_j = torch.randint(board.shape[1] - self.dataset.abs_dim + 1, (1,)).item()
                        i_s, i_e = patch_pos_i, patch_pos_i + self.dataset.abs_dim
                        j_s, j_e = patch_pos_j, patch_pos_j + self.dataset.abs_dim
                        masked_board[i,i_s:i_e,j_s:j_e] = self.masking_token
            
        if bool_report:
            stats = {
                'boardType': stats_boardType
            }
            return masked_board, stats
        else:
            return masked_board
    
    #.......................................
    # training the network
    #.......................................
    
    def train(self, bool_wandb = False):
        
        """
        train the transformer
        
        Args:
            bool_wandb (bool): whether or not to log stats to wandb
        Returns:
            cum_losses (np.ndarray): vector of cummulative losses over training
        """
            
        # make dataloader from dataset
        self.dataset.set_type = 'train'
        dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)

        # print details about training
        print(f'training transformer ({self.transformer_type}) @ {self.run_name}')
        print(f'\n* Architecture:')
        print(f'... {self.n_embd}-D embeddings')
        print(f'... {self.n_layer} layers')
        print(f'... {self.n_head} attn heads, each {self.n_embd // self.n_head} dim')
        print(f'\n* Training:')
        print(f'... {self.epochs} epochs, at {len(dataloader)} steps/epoch')
        
        # init counters and containers
        self.step = 0
        self.epoch = 0
        start_time = time.time()

        cum_losses = []
        log = { # to log stats about training
            "written":np.zeros(self.epochs),
            "epoch": np.zeros(self.epochs), 
            "loss": np.zeros(self.epochs), 
            "error": np.zeros(self.epochs),
            "nb_boards": np.zeros(self.epochs),
            "nb_boards_singleRoot": np.zeros(self.epochs),
            "nb_boards_doubleRoot": np.zeros(self.epochs),
            "nb_boards_composite": np.zeros(self.epochs),
            "train_acc_tokens": np.zeros(self.epochs),
            "train_acc_boards": np.zeros(self.epochs),
            "train_breakdown_0": np.zeros(self.epochs),
            "train_breakdown_1": np.zeros(self.epochs),
            "train_breakdown_2": np.zeros(self.epochs),
            "test_acc_tokens": np.zeros(self.epochs),
            "test_acc_boards": np.zeros(self.epochs),
            "test_breakdown_0": np.zeros(self.epochs),
            "test_breakdown_1": np.zeros(self.epochs),
            "test_breakdown_2": np.zeros(self.epochs)
        }

        # train        
        for epoch in range(self.epochs):
            
            losses = [] # to store losses
            percent_masked = [] # to store percent of masked tokens in each batch
            percent_error = [] # to store percent of reconstrunction error in each batch

            # change dataset parameters to implement curriculum
            # note: requires reinitializing dataloader to take into account new board_types_proba
            if self.board_curriculum:
                self.dataset.board_types_proba = ramp(epoch, np.array(self.board_proba_start), 
                                                    np.array(self.board_proba_end), 
                                                    self.board_proba_epoch_start, 
                                                    self.board_proba_epoch_end)
                dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)
            
            for _, boards in enumerate(dataloader):
                    
                self.optim.zero_grad()

                # make sure boards are on the right device
                boards = boards.to(device=self.device)

                # compute masking probability based on epoch
                masking_p = max(self.masking_p * min(1., epoch/self.masking_patch_epoch), 0.01)
                
                # get masked boards
                masked_boards = self._mask_input(boards, masking_p, 
                                                 patch_masking = epoch >= self.masking_patch_epoch, 
                                                 patch_masking_p = self.masking_patch_p)
                
                # reshape for the forward pass
                boards = rearrange(boards, 'b h w -> b (h w)')
                masked_boards = rearrange(masked_boards, 'b h w -> b (h w)')
                
                # forward
                logits = self.transformer(masked_boards)
                
                # compute loss and gradients
                loss = F.cross_entropy(rearrange(logits, 'b l v -> (b l) v'), rearrange(boards, 'b l -> (b l)'))
                loss.backward()

                # update weights
                self.optim.step()

                # log stats
                losses.append(loss.cpu().detach().item())
                percent_masked.append(1. - (torch.sum(boards == masked_boards)/torch.numel(boards)).cpu().detach().item())
                reconstructed_boards = torch.argmax(logits, dim = -1)
                error = torch.mean((reconstructed_boards != boards).float())
                percent_error.append(error.cpu().detach().item())

                self.step += 1
                self.epoch = epoch
            
            # aggregate stats over batch
            cum_loss = np.sum(np.array(losses))
            cum_losses.append(cum_loss)
            avg_error = np.mean(np.array(percent_error))
            
            # report on loss
            if epoch % self.epoch2print == 0:
                speed = (time.time() - start_time) / (epoch + 1)
                print(f'\ntraining transformer ({self.transformer_type}) @ {self.run_name}')
                print(f'... epoch {epoch}:\tcumloss = {np.round(cum_loss, 4)}\t{speed:0.4f} s/epoch')
                # print average percent masked
                print(f'...... percent masked = {np.mean(np.array(percent_masked)):.4f}, patch masking = {epoch >= self.masking_patch_epoch}')
                print(f'...... avg. percent error = {100 * avg_error:0.4f}')
                print(f'...... dataset board proba: {self.dataset.board_types_proba}')

            # test prediction with current model on both training and test sets
            if epoch % self.epoch2test == 0 or (epoch + 1) == self.epochs:

                print('>>> computing accuracies on training and test sets...')

                # increase set sizes by having different masking rounds on the same board
                masking_round = 2

                # find the distribution of board types in the test set
                # we aim to have the same distribution in the training set as the error varies significantly between board types
                # we cap the number of boards to 100 for each board type
                if self.dataset.sets['singleRoot_test'].shape[0] == 0: # in the case where the dataset does not have a test set
                    nb_board_singleRoot = 100
                    nb_board_doubleRoot = 100
                    nb_board_composite = 100
                else:
                    nb_board_singleRoot = min(100, self.dataset.sets['singleRoot_test'].shape[0])
                    nb_board_doubleRoot = min(100, self.dataset.sets['doubleRoot_test'].shape[0])
                    nb_board_composite = min(100, self.dataset.sets['singleComposite_test'].shape[0])
                nb_training_boards = masking_round * (nb_board_singleRoot + nb_board_doubleRoot + nb_board_composite)

                # transiently change board_types_proba to get a distribution of boards that matches the test set
                board_types_proba = np.array([nb_board_singleRoot,
                                                           nb_board_doubleRoot,
                                                           nb_board_composite])
                board_types_proba = board_types_proba / np.sum(board_types_proba)
                board_types_proba_before = np.copy(self.dataset.board_types_proba)
                self.dataset.board_types_proba = board_types_proba

                # get accuracy on training set, sampling board types with the same distribution as the test set
                train_acc_tokens, train_acc_boards, _, _, _, train_acc_boards_breakdown = self.test_model_accuracy(nb_training_boards, 
                                                    masking_p = masking_p, 
                                                    patch_masking = epoch >= self.masking_patch_epoch, 
                                                    patch_masking_p = self.masking_patch_p, 
                                                    verbose = False,
                                                    display_erroneous_boards = False)
                
                # reset board_types_proba to what it was before
                self.dataset.board_types_proba = board_types_proba_before
                
                # check that test set is not empty
                if self.dataset.sets['singleRoot_test'].shape[0] == 0:
                    test_acc_tokens = -1
                    test_acc_boards = -1
                    test_acc_boards_breakdown = [-1, -1, -1]
                else:
                    # get accuracy on test set
                    test_acc_tokens, test_acc_boards, _, _, _, test_acc_boards_breakdown = self.test_model_accuracy_set(target_set = 'test', 
                                                        masking_p = masking_p, 
                                                        patch_masking = epoch >= self.masking_patch_epoch, 
                                                        patch_masking_p = self.masking_patch_p,
                                                        masking_rounds=masking_round, 
                                                        verbose = False,
                                                        display_erroneous_boards = False,
                                                        count_max = 100) # we cap the number of boards to 100 for each board type
                
                # log
                log["written"][epoch] = 1
                log["epoch"][epoch] = epoch
                log["loss"][epoch] = cum_loss
                log["error"][epoch] = avg_error
                log["nb_boards"][epoch] = nb_training_boards
                log["nb_boards_singleRoot"][epoch] = nb_board_singleRoot
                log["nb_boards_doubleRoot"][epoch] = nb_board_doubleRoot
                log["nb_boards_composite"][epoch] = nb_board_composite
                log["train_acc_tokens"][epoch] = train_acc_tokens
                log["train_acc_boards"][epoch] = train_acc_boards
                log["train_breakdown_0"][epoch] = train_acc_boards_breakdown[0]
                log["train_breakdown_1"][epoch] = train_acc_boards_breakdown[1]
                log["train_breakdown_2"][epoch] = train_acc_boards_breakdown[2]
                log["test_acc_tokens"][epoch] = test_acc_tokens
                log["test_acc_boards"][epoch] = test_acc_boards
                log["test_breakdown_0"][epoch] = test_acc_boards_breakdown[0]
                log["test_breakdown_1"][epoch] = test_acc_boards_breakdown[1]
                log["test_breakdown_2"][epoch] = test_acc_boards_breakdown[2]

                if bool_wandb: # report to wandb
                    wandb.log({
                        "epoch": epoch, 
                        "loss": cum_loss, 
                        "error": 100 * avg_error,
                        "nb_boards": nb_training_boards,
                        "train_acc_tokens": train_acc_tokens,
                        "train_acc_boards": train_acc_boards,
                        "train_breakdown_0": train_acc_boards_breakdown[0],
                        "train_breakdown_1": train_acc_boards_breakdown[1],
                        "train_breakdown_2": train_acc_boards_breakdown[2],
                        "test_acc_tokens": test_acc_tokens,
                        "test_acc_boards": test_acc_boards,
                        "test_breakdown_0": test_acc_boards_breakdown[0],
                        "test_breakdown_1": test_acc_boards_breakdown[1],
                        "test_breakdown_2": test_acc_boards_breakdown[2]
                        }, step=self.step)

                # print meta info
                print('*** testing performance on training/test set at epoch {}'.format(epoch))
                print(f'... nb_board_singleRoot: {nb_board_singleRoot}')
                print(f'... nb_board_doubleRoot: {nb_board_doubleRoot}')
                print(f'... nb_board_composite: {nb_board_composite}')
                print(f'... masking probability: {masking_p}')
                print(f'... patch masking: {epoch >= self.masking_patch_epoch}')
                print(f'... patch masking probability: {self.masking_patch_p}')
                # print accuracies
                print(f'... train_acc_tokens = {train_acc_tokens:.4f}\ttrain_acc_boards = {train_acc_boards:.4f}')
                print(f'... train_acc_boards_breakdown = {train_acc_boards_breakdown}')
                print(f'... test_acc_tokens = {test_acc_tokens:.4f}\ttest_acc_boards = {test_acc_boards:.4f}')
                print(f'... test_acc_boards_breakdown = {test_acc_boards_breakdown}')

            # save current version of the network (save on the last epoch)
            if self.save_model and (epoch % self.epoch2save == 0 or (epoch + 1) == self.epochs) and not bool_wandb:
                torch.save(self.transformer.state_dict(), f'./runs/{self.run_name}/checkpoints/model_{epoch}.pth')
                
        # print total duration of training
        total_time = time.time()-start_time
        total_time_min = total_time // 60
        total_time_sec = total_time - total_time_min * 60
        print(f'\n>> training completed in {total_time_min} min, {total_time_sec:.3f} s')

        # check to see if anything was written in log and if so, save it to disk
        if np.sum(log["written"]) > 0 and not bool_wandb:
            # write to disck with pickle
            with open(f'./runs/{self.run_name}/run_log.pkl', 'wb') as f:
                pickle.dump(log, f)
        
        return np.array(cum_losses)
    
    #..................................
    # compute accuracy
    #..................................
    
    # look at reconstruction error on a random set of masked boards from the training set
    @torch.no_grad()
    def test_model_accuracy(self, nb_instances, masking_p, patch_masking = False, 
                            patch_masking_p = 0.1, verbose = False, display_erroneous_boards = False):
        
        """
        test model accuracy on a given set of boards (splits predefined in dataset)

        Args:
            nb_instances (int): number of boards to test on
            masking_p (float): percentage of tokens to mask
            patch_masking (bool): if True, mask patches of tokens instead of individual tokens
            patch_masking_p (float): probability of patch masking
            verbose (bool): whether or not to print info/stats
            display_erroneous_boards (bool): whether or not to display boards that are not reconstructed properly

        Returns:
            acc_tokens (float): accuracy across tokens
            acc_boards (float): accuracy across boards
            boards (np.ndarray): original boards
            rec_boards (np.ndarray): reconstructed boards
            masks (np.ndarray): masks used to mask boards
            acc_boards_breakdown (list of float): accuracy across boards, broken down by board type
        """

        # get n boards from the training set
        boards, board_types = self.dataset.get_n_training_boards(nb_instances)

        # convert to torch tensor
        boards = torch.from_numpy(boards).long().to(self.device)
        if verbose: print(f'boards.shape = {boards.shape}')

        # random masking
        masked_boards = self._mask_input(boards, 
                                         masking_p, 
                                         patch_masking = patch_masking, 
                                         patch_masking_p = patch_masking_p)
        if verbose: 
            print(f'masked_boards.shape = {masked_boards.shape}')
            print(f'percentage masked = {torch.sum(masked_boards == self.masking_token).item() / masked_boards.numel():.3f}')
        
        # reshape inputs, pass through transformer, get logits  
        boards = rearrange(boards, 'b h w -> b (h w)')
        masked_boards = rearrange(masked_boards, 'b h w -> b (h w)')
        print(f'test_model_accuracy: boards shape is {boards.size()}')
        logits = self.transformer(masked_boards)

        # get reconstructed boards
        rec_boards = torch.argmax(logits, dim = -1)

        # compute overall accuracy
        error_token = boards != rec_boards
        error_board = torch.sum(error_token, dim = -1) > 0
        
        # find out which board types are erroneous
        error_board_types = board_types[error_board.cpu().detach().numpy()]
        acc_tokens = 1. - torch.sum(error_token).item() / boards.numel()
        acc_boards = 1. - torch.sum(error_board).item() / boards.size()[0]
        acc_boards_breakdown  = [
            1. - np.sum(error_board_types == 0) / np.sum(board_types == 0),
            1. - np.sum(error_board_types == 1) / np.sum(board_types == 1),
            1. - np.sum(error_board_types == 2) / np.sum(board_types == 2)
        ]

        # print stats
        if verbose:
            print(f'accuracy across tokens = {acc_tokens:.3f}')
            print(f'accuracy across boards = {acc_boards:.3f}')
            print(f'... single root: {np.sum(error_board_types == 0)}/{np.sum(board_types == 0)}, {acc_boards_breakdown[0]:.3f} acc')
            print(f'... double root: {np.sum(error_board_types == 1)}/{np.sum(board_types == 1)}, {acc_boards_breakdown[1]:.3f} acc')
            print(f'... composite {np.sum(error_board_types == 2)}/{np.sum(board_types == 2)}, {acc_boards_breakdown[2]:.3f} acc')
        
        # convert boards and masks to numpy
        boards = rearrange(boards, 'b (h w) -> b h w', h = self.dataset.board_dim).cpu().detach().numpy()
        rec_boards = rearrange(rec_boards, 'b (h w) -> b h w', h = self.dataset.board_dim).cpu().detach().numpy()
        masks = torch.zeros_like(masked_boards).int()
        masks[masked_boards == self.masking_token] = 1
        masks = rearrange(masks, 'b (h w) -> b h w', h = self.dataset.board_dim).cpu().detach().numpy()

        # render boards with error to terminal
        if display_erroneous_boards:
            for idx_board in range(boards.shape[0]):
                curr_error = np.sum(boards[idx_board] != rec_boards[idx_board])
                if curr_error > 0:
                    board = boards[idx_board]
                    mask = masks[idx_board]
                    board_abs = -1 * np.ones_like(board).astype(np.int)
                    board_abs = np.where(board >= self.dataset.vocab_bck, 0, board_abs)

                    rec_board = rec_boards[idx_board]
                    rec_board_abs = -1 * np.ones_like(rec_board).astype(np.int)
                    rec_board_abs = np.where(rec_board >= self.dataset.vocab_bck, 0, rec_board_abs)

                    print('-'*50,f'\nboard {idx_board}: error = {curr_error}')
                    print('original:')
                    self.dataset.render_board(board, board_abs, mask=mask, figsize=(3,3), label_size=(8,0.15))
                    print('reconstructed:')
                    self.dataset.render_board(rec_board, rec_board_abs, mask=np.zeros_like(board), figsize=(3,3), label_size=(8,0.15))

        return acc_tokens, acc_boards, boards, rec_boards, masks, acc_boards_breakdown

    # look at reconstruction error on a random set of masked boards from the specficied set (train, val, test)
    @torch.no_grad()
    def test_model_accuracy_set(self, target_set, masking_p, patch_masking = False, 
                            patch_masking_p = 0.1, masking_rounds = 1, verbose = False, display_erroneous_boards = False, count_max = 100):
        
        """
        test model accuracy on a given set of boards (splits predefined in dataset)

        Args:
        - target_set: 'train', 'val', 'test', set to test on
        - masking_p: percentage of tokens to mask
        - patch_masking: if True, mask patches of tokens instead of individual tokens
        - patch_masking_p: percentage of patches to mask
        - masking_rounds: number of times to mask each board
        - verbose: whether or not to print info/stats
        - display_erroneous_boards: whether or not to display boards that are not reconstructed properly
        - count_max: maximum number of boards to test on for each board type

        Returns:
        - acc_tokens: accuracy across tokens
        - acc_boards: accuracy across boards
        - boards: original boards
        - rec_boards: reconstructed boards
        - masks: masks used to mask boards
        - acc_boards_breakdown: accuracy across boards, broken down by board type
        """

        # make boards
        if masking_rounds > 1:
            boards = []
            board_types = []
            for _ in range(masking_rounds):
                boards_, board_types_ = self.dataset.get_all_boards(target_set, count_max = count_max)
                boards.append(boards_)
                board_types.append(board_types_)
            boards = np.concatenate(boards, axis = 0)
            board_types = np.concatenate(board_types, axis = 0)
        else:
            boards, board_types = self.dataset.get_all_boards(target_set, count_max = count_max)

        # convert to torch tensor
        boards = torch.from_numpy(boards).long().to(self.device)
        if verbose: print(f'boards.shape = {boards.shape}')

        # random masking
        masked_boards = self._mask_input(boards, 
                                         masking_p, 
                                         patch_masking = patch_masking, 
                                         patch_masking_p = patch_masking_p)
        if verbose: 
            print(f'masked_boards.shape = {masked_boards.shape}')
            print(f'percentage masked = {torch.sum(masked_boards == self.masking_token).item() / masked_boards.numel():.3f}')
        
        # reshape and forward  
        boards = rearrange(boards, 'b h w -> b (h w)')
        masked_boards = rearrange(masked_boards, 'b h w -> b (h w)')

        print(f'test_model_accuracy_set: boards shape is {boards.size()}')
        logits = self.transformer(masked_boards)

        # get reconstructed boards
        rec_boards = torch.argmax(logits, dim = -1)

        # compute overall accuracy
        error_token = boards != rec_boards
        error_board = torch.sum(error_token, dim = -1) > 0
        # find out which board types are erroneous
        error_board_types = board_types[error_board.cpu().detach().numpy()]
        acc_boards_breakdown  = [
            1. - np.sum(error_board_types == 0) / np.sum(board_types == 0),
            1. - np.sum(error_board_types == 1) / np.sum(board_types == 1),
            1. - np.sum(error_board_types == 2) / np.sum(board_types == 2)
        ]

        acc_tokens = 1. - torch.sum(error_token).item() / boards.numel()
        acc_boards = 1. - torch.sum(error_board).item() / boards.size()[0]
        if verbose:
            print(f'accuracy across tokens = {acc_tokens:.3f}')
            print(f'accuracy across boards = {acc_boards:.3f}')
            print(f'... single root: {np.sum(error_board_types == 0)}/{np.sum(board_types == 0)}, {acc_boards_breakdown[0]:.3f} acc')
            print(f'... double root: {np.sum(error_board_types == 1)}/{np.sum(board_types == 1)}, {acc_boards_breakdown[1]:.3f} acc')
            print(f'... composite {np.sum(error_board_types == 2)}/{np.sum(board_types == 2)}, {acc_boards_breakdown[2]:.3f} acc')
        
        # convert boards and masks to numpy
        boards = rearrange(boards, 'b (h w) -> b h w', h = self.dataset.board_dim).cpu().detach().numpy()
        rec_boards = rearrange(rec_boards, 'b (h w) -> b h w', h = self.dataset.board_dim).cpu().detach().numpy()
        masks = torch.zeros_like(masked_boards).int()
        masks[masked_boards == self.masking_token] = 1
        masks = rearrange(masks, 'b (h w) -> b h w', h = self.dataset.board_dim).cpu().detach().numpy()

        # render boards with error
        if display_erroneous_boards:
            for idx_board in range(boards.shape[0]):
                curr_error = np.sum(boards[idx_board] != rec_boards[idx_board])
                if curr_error > 0:
                    board = boards[idx_board]
                    mask = masks[idx_board]
                    board_abs = -1 * np.ones_like(board).astype(np.int)
                    board_abs = np.where(board >= self.dataset.vocab_bck, 0, board_abs)

                    rec_board = rec_boards[idx_board]
                    rec_board_abs = -1 * np.ones_like(rec_board).astype(np.int)
                    rec_board_abs = np.where(rec_board >= self.dataset.vocab_bck, 0, rec_board_abs)

                    print('-'*50,f'\nboard {idx_board}: error = {curr_error}')
                    print('original:')
                    self.dataset.render_board(board, board_abs, mask=mask, figsize=(3,3), label_size=(8,0.15))
                    print('reconstructed:')
                    self.dataset.render_board(rec_board, rec_board_abs, mask=np.zeros_like(board), figsize=(3,3), label_size=(8,0.15))

        return acc_tokens, acc_boards, boards, rec_boards, masks, acc_boards_breakdown

    # look at reconstruction error on a specific set of masked boards
    @torch.no_grad()
    def test_model_accuracy_subset(self, boards, masks, bool_show_erroneous_boards = False):
        
        """
        test model accuracy on specified boards

        Args:
            boards (np.ndarray): boards to test on
            masks (np.ndarray): masks used to mask boards
            bool_show_erroneous_boards (bool): whether or not to display boards that are not reconstructed properly
        
        Returns:
            report (dict): Dictionary containing the following keys:
                - boards_recons (np.ndarray): reconstructed boards
                - boards (np.ndarray): original boards
                - masks (np.ndarray): masks used to mask boards
                - error_token (float): error across tokens
                - error_masked_token (float): error across masked tokens
                - error_board (float): error across boards
                - erroneous_boards (np.ndarray): indices of boards that were not reconstructed properly
        """

        batch, h, w = boards.shape

        #..................................
        # mask board
        #..................................
        boards = torch.from_numpy(rearrange(boards,'b h w -> b (h w)')).long().to(self.device) # convert to tensor (k,n*n)
        masks = torch.from_numpy(rearrange(masks,'b h w -> b (h w)')).long().to(self.device) # convert to tensor (1,n*n)

        fully_masked_boards = self.masking_token * torch.ones(boards.shape, device = self.device)
        masked_boards = (masks * fully_masked_boards + (1-masks) * boards).long() # should be indices

        #..................................
        # predict
        #..................................
        logits = self.transformer(masked_boards) # (b,n*n,v)
        boards_recons = torch.argmax(F.softmax(logits, dim = -1), dim = -1)

        #..................................
        # convert all to numpy
        #..................................
        boards = rearrange(boards, 'b (h w) -> b h w', h = h).cpu().detach().numpy()
        masks = rearrange(masks, 'b (h w) -> b h w', h = h).cpu().detach().numpy()
        masked_boards = rearrange(masked_boards, 'b (h w) -> b h w', h = h).cpu().detach().numpy()
        boards_recons = rearrange(boards_recons, 'b (h w) -> b h w', h = h).cpu().detach().numpy()

        #..................................
        # compute cumulative error between original and reconstructed board
        #..................................
        error_token = np.sum(np.abs(boards - boards_recons) > 0) / boards.size
        error_masked_token = np.sum(np.abs(boards[masks == 1] - boards_recons[masks == 1]) > 0) / np.sum(masks)

        #..................................
        # compute board error
        #..................................
        erroneous_boards = np.where(np.sum(np.abs(boards-boards_recons) > 0, axis = (1,2)) > 0)[0]
        error_board = len(erroneous_boards) / boards.shape[0]

        if bool_show_erroneous_boards:
            for i in erroneous_boards:
                print('*' * 50)
                print(boards[i])
                print('.'*25)
                print(masked_boards[i])
                print('.'*25)
                print(boards_recons[i])
                print('.'*25)
                # print delta between boards
                print(-1 * (np.abs(boards_recons[i] - boards[i]) > 0).astype(int))

        report = {
            'boards_recons': boards_recons,
            'boards': boards,
            'masks': masks,
            'error_token': error_token,
            'error_masked_token': error_masked_token,
            'error_board': error_board,
            'erroneous_boards': erroneous_boards,
        }

        return report

    #.......................................
    # probing the network
    #.......................................
    
    # visualize learned token and position embeddings
    def analyze_embeddings_tokenandpos(self):

        """
        make summary plots for the token embeddings and positional encoding learned by the model.
        Produces 3 plots:
        * Plot 1, 4 subplots:
            - token embeddings
            - euclidian distance between token embeddings
            - euclidian distance between token embeddings with unit norm
            - cosine distance between token embeddings
        * Plot 2, 2 subplots:
            - position embeddings
            - euclidian distance between position embeddings
        * Plot 3, 1 subplot:
            - euclidian distance between position embeddings, rearranged to show the structure of the board
        /!\ This function assumes a specifc folder architecture (see savefig function)
        """
        
        #..................................
        # plot (1) token embeddings
        #..................................
        fig, axes = plt.subplots(2, 2, figsize = (2*3, 1*3), dpi = 300)
        
        token_emb = self.transformer.tok_emb.weight.cpu().detach().numpy()
        im00 = axes[0,0].imshow(token_emb, cmap = 'magma')
        axes[0,0].axhline(y = self.dataset.vocab_bck-0.5, color = 'w', linestyle = '-', linewidth = 0.5)
        axes[0,0].axhline(y = self.dataset.vocab_bck + self.dataset.vocab_token -0.5, color = 'w', linestyle = '-', linewidth = 0.5)
        axes[0,0].title.set_text('token embeddings')
        plt.colorbar(im00, ax=axes[0,0])

        im10 = axes[1,0].imshow(dist(token_emb), cmap = 'magma')
        # add horizontal and vertical lines
        axes[1,0].axhline(y = self.dataset.vocab_bck-0.5, color = 'w', linestyle = '-', linewidth = 0.5)
        axes[1,0].axhline(y = self.dataset.vocab_bck + self.dataset.vocab_token -0.5, color = 'w', linestyle = '-', linewidth = 0.5)
        axes[1,0].axvline(x = self.dataset.vocab_bck-0.5, color = 'w', linestyle = '-', linewidth = 0.5)
        axes[1,0].axvline(x = self.dataset.vocab_bck + self.dataset.vocab_token -0.5, color = 'w', linestyle = '-', linewidth = 0.5)
        axes[1,0].title.set_text('euclidian')
        plt.colorbar(im10, ax=axes[1,0])

        im11 = axes[1,1].imshow(dist_unit(token_emb), cmap = 'magma')
        # add horizontal and vertical lines
        axes[1,1].axhline(y = self.dataset.vocab_bck-0.5, color = 'w', linestyle = '-', linewidth = 0.5)
        axes[1,1].axhline(y = self.dataset.vocab_bck + self.dataset.vocab_token -0.5, color = 'w', linestyle = '-', linewidth = 0.5)
        axes[1,1].axvline(x = self.dataset.vocab_bck-0.5, color = 'w', linestyle = '-', linewidth = 0.5)
        axes[1,1].axvline(x = self.dataset.vocab_bck + self.dataset.vocab_token -0.5, color = 'w', linestyle = '-', linewidth = 0.5)
        axes[1,1].title.set_text('euclidian unit norm')
        plt.colorbar(im11, ax=axes[1,1])

        im01 = axes[0,1].imshow(dist_cosine(token_emb), cmap = 'magma')
        # add horizontal and vertical lines
        axes[0,1].axhline(y = self.dataset.vocab_bck-0.5, color = 'w', linestyle = '-', linewidth = 0.5)
        axes[0,1].axhline(y = self.dataset.vocab_bck + self.dataset.vocab_token -0.5, color = 'w', linestyle = '-', linewidth = 0.5)
        axes[0,1].axvline(x = self.dataset.vocab_bck-0.5, color = 'w', linestyle = '-', linewidth = 0.5)
        axes[0,1].axvline(x = self.dataset.vocab_bck + self.dataset.vocab_token -0.5, color = 'w', linestyle = '-', linewidth = 0.5)
        axes[0,1].title.set_text('cosine')
        plt.colorbar(im01, ax=axes[0,1])

        plt.tight_layout()
        plt.savefig(f'./runs/{self.run_name}/analysis/embedding_token.jpg', bbox_inches='tight', dpi=300)
        plt.close()

        #..................................
        # plot (2) positional encoding
        #..................................
        pos_emb = self.transformer.pos_emb.cpu().detach().numpy()[0]
        
        fig, axes = plt.subplots(1, 2, figsize = (2*3, 1*3), dpi = 300)
        axes[0].imshow(pos_emb, cmap = 'magma')
        axes[0].title.set_text('position embeddings')
        im1 = axes[1].imshow(dist(pos_emb), cmap = 'magma')
        axes[1].set(ylabel=f'euclidean dist.')
        plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.savefig(f'./runs/{self.run_name}/analysis/embedding_pos.jpg', bbox_inches='tight', dpi=300)
        plt.close()

        #..................................
        # plot (3) positional encoding rearranged to show board structure
        #..................................
        D = dist(pos_emb)
        # make it symetric
        D = D + D.T
        # rearrange
        D = rearrange(D, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', 
                      h1 = self.dataset.board_dim,
                      h2 = self.dataset.board_dim)
        print(f'D.shape: {D.shape}')

        fig, axes = plt.subplots(D.shape[0], D.shape[1], figsize = (D.shape[0]*3, D.shape[1]*3))
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                axes[i,j].imshow(D[i,j], cmap = 'magma')
                axes[i,j].set(xticks=[], yticks=[])

        plt.savefig(f'./runs/{self.run_name}/analysis/embedding_pos_2d.jpg', bbox_inches='tight', dpi=300)
        plt.close()
    
    # visualize token embeddings for a single board as they evolve through the transformer
    @torch.no_grad()
    def analyze_embeddings(self, 
                           board = None,
                           board_abs = None,
                           board_params = {'nb_root_abs':1, 'composite': None, 'margin': 0}, 
                           masking = {'bool_mask':True, 'p':0.25}, 
                           bool_sharedCmap = False):

        """
        Analysis of token embeddings (z) throughout the transformer for a single board.
        Produces 2 plots:
        * Plot 1: z across blocks (row) and subblocks (columns)
        * Plot 2: distance between different token embeddings within subblocks
        * Plot 3: distance between same token embeddings across subblocks
        /!\ This function assumes a specifc folder architecture (see savefig function)

        Args:
        - board: (h,w) numpy array of tokens
        - board_abs: (h,w) numpy array of abstraction indices
        - board_params: dict with parameters to generate the board
        - masking: dict with parameters to mask the board
        - bool_sharedCmap: if True, use the same colorbar for all plots

        Returns:
            None
        """
        
        #..................................
        # make board
        #..................................

        if board is None or board_abs is None:
            board, board_abs = self.dataset._draw_board(nb_root_abs = board_params['nb_root_abs'], 
                                                 composite = board_params['composite'],
                                                 margin = board_params['margin'])
        h, w = board.shape

        #..................................
        # mask board
        #..................................

        board = torch.from_numpy(rearrange(board,'h w -> 1 (h w)')).long().to(self.device) # convert to tensor (1,n*n)

        if masking['bool_mask']:
            fully_masked_board = self.masking_token * torch.ones(board.shape, device = self.device)
            mask = torch.bernoulli(masking['p'] * torch.ones(board.shape, device = self.device))
            masked_board = (mask * fully_masked_board + (1-mask) * board).long() # should be indices

        else:
            masked_board = board

        #..................................
        # register hooks & get activations
        #..................................
        self._register_hooks_z()
        self.activations_z = {} # empty dict to store activations
        logits = self.transformer(masked_board) # (b,n*n,v)
        self._unregister_hooks() # remove hooks

        # reconstruct board from logits
        board_reconstructed = torch.argmax(F.softmax(logits, dim = -1), dim = -1)

        #..................................
        # convert all to numpy
        #..................................
        board = rearrange(board[0], '(h w) -> h w', h = h).cpu().detach().numpy()
        masked_board = rearrange(masked_board[0], '(h w) -> h w', h = h).cpu().detach().numpy()
        board_reconstructed = rearrange(board_reconstructed[0], '(h w) -> h w', h = h).cpu().detach().numpy()

        # show board, masked board and reconstructed board
        print(f'board:\n{board}')
        print(f'masked_board:\n{masked_board}')
        print(f'board recons:\n{board_reconstructed}')
        
        # go through activation and find min/max values for normalization across plots
        activations = {}
        act_min = np.Inf # for normalization across plots
        act_max = -np.Inf # for normalization across plots

        for k, v in self.activations_z.items():
            #print(f'{k}: {v.size()}')
            activations[k] = v[0].cpu().detach().numpy() # remove batch dimension
            act_min = np.minimum(act_min, np.min(activations[k]))
            act_max = np.maximum(act_max, np.max(activations[k]))
        
        #..................................
        # plot (1) shows z across blocks (row) and subblocks (columns)
        print('plot 1: z across blocks and sub-blocks')
        #..................................
        
        populations = ['z', 'attn', 'z_attn', 'mlp', 'z_attn_mlp']

        fig, axes = plt.subplots(self.n_layer, len(populations), figsize = (len(populations)*3, self.n_layer*3))

        for idx_block in range(self.n_layer):
            for idx_pop in range(len(populations)):
                
                z = activations[f'b_{idx_block}_{populations[idx_pop]}']

                if bool_sharedCmap:
                    axes[idx_block, idx_pop].imshow(z, cmap = 'magma', vmin=act_min, vmax=act_max)
                else:
                    axes[idx_block, idx_pop].imshow(z, cmap = 'magma')

                if idx_block == 0:
                    axes[idx_block, idx_pop].title.set_text(populations[idx_pop])
                if idx_pop == 0:
                    axes[idx_block, idx_pop].set(ylabel=f'block {idx_block}')

        plt.savefig(f'./runs/{self.run_name}/analysis/embeddings_z.jpg', bbox_inches='tight', dpi=self.plot_dpi)
        plt.close()

        #..................................
        # plot (2) distances between zs within subblocks
        print('plot 2: distance between z within sub-blocks')
        #..................................

        fig, axes = plt.subplots(self.n_layer, len(populations), figsize = (len(populations)*3, self.n_layer*3))
        imgs = []

        for idx_block in range(self.n_layer):
            for idx_pop in range(len(populations)):
                
                z = activations[f'b_{idx_block}_{populations[idx_pop]}']
                z_plot = dist(z)

                # add abstraction information
                # get all abstractions by looking at board_abs
                abs_indices = np.unique(board_abs)
                abs_indices = abs_indices[abs_indices > -1] # remove background
                z_abs = np.linspace(0., np.max(z_plot), len(abs_indices)+1)[1:] # plot intensity for each abs

                board_abs_flat = rearrange(board_abs, 'h w -> (h w)')
                assert board_abs_flat.shape[0] == z_plot.shape[0], "dimension mismatch"

                for i in range(z_plot.shape[0]):
                    if board_abs_flat[i] > -1: # token belongs to abstraction
                        j = np.where(abs_indices == board_abs_flat[i])[0][0]
                        z_plot[i,:i+1] = z_abs[j]


                imgs.append(axes[idx_block, idx_pop].imshow(z_plot, cmap = 'magma'))
                plt.colorbar(imgs[-1], ax=axes[idx_block, idx_pop])

                if idx_block == 0:
                    axes[idx_block, idx_pop].title.set_text(populations[idx_pop])
                if idx_pop == 0:
                    axes[idx_block, idx_pop].set(ylabel=f'block {idx_block}')

        plt.savefig(f'./runs/{self.run_name}/analysis/embeddings_z_dist.jpg', bbox_inches='tight', dpi=self.plot_dpi)
        plt.close()
        
        #..................................
        # plot (3) distances between same z across subblocks
        print('plot 3: distances between same z across sub-blocks')
        #..................................

        # add abstraction information
        # get all abstractions by looking at board_abs
        abs_indices = np.unique(board_abs)
        abs_indices = abs_indices[abs_indices > -1] # remove background
        plot_colors = ['tab:'+ c for c in ['blue','orange','green','red','purple','brown','pink','olive']]

        n, d = activations[f'b_{0}_{populations[0]}'].shape
        Z = np.zeros((self.n_layer * len(populations), n, d))
        board_abs_flat = rearrange(board_abs, 'h w -> (h w)').astype(int)

        for idx_block in range(self.n_layer):
            for idx_pop in range(len(populations)):
                i = idx_block * len(populations) + idx_pop
                Z[i,:,:] = activations[f'b_{idx_block}_{populations[idx_pop]}']
        
        # make a square plot m*m ~ n 
        m1, m2 = find_best_plot_dim(n)

        fig, axes = plt.subplots(m1, m2, figsize = (m2*2, m1*2))
        imgs = []

        for idx_plot in range(n):
            
            i = idx_plot // m1
            j = idx_plot - i * m1

            z_plot = dist(Z[:,idx_plot,:]) # (l,d) -> (l,l)
            imgs.append(axes[i, j].imshow(z_plot, cmap = 'magma'))

            for idx_block in range(self.n_layer):
                axes[i, j].axhline(y = idx_block * len(populations)-0.5, color = 'w', linestyle = '-', linewidth = 0.5)
                axes[i, j].axvline(x = idx_block * len(populations)-0.5, color = 'w', linestyle = '-', linewidth = 0.5)

            plt.colorbar(imgs[-1], ax=axes[i, j])
            if board_abs_flat[idx_plot] > -1:
                axes[i, j].title.set_text(f't-{idx_plot}*{board_abs_flat[idx_plot]}')
                k = Z.shape[0] // 5
                l = np.where(abs_indices == board_abs_flat[idx_plot])[0][0]
                rect = Rectangle((-0.5,Z.shape[0]-k-0.5),k,k,linewidth=2,edgecolor='none',facecolor= plot_colors[l])
                axes[i,j].add_patch(rect)
            else:
                axes[i, j].title.set_text(f't-{idx_plot}')
            axes[i,j].get_xaxis().set_visible(False)
            axes[i,j].get_yaxis().set_visible(False)

        plt.savefig(f'./runs/{self.run_name}/analysis/embeddings_z_dist_across.jpg', bbox_inches='tight', dpi=self.plot_dpi)
        plt.close()
    
    # sub-function to `analyze_all`
    def analyze_all_z(self, board_abs, folder = 'default'):

        """
        Analysis of token embeddings (z) throughout the transformer for a single board.
        Will plot the following:
        * Plot 1: z across blocks (row) and subblocks (columns)
        * Plot 2: norm(z) across blocks (row) and subblocks (columns), arrange to show board structure
        * Plot 3.1 & 3.2: eucledian distance between different token embeddings within subblocks.
        * Plot 4: eucledian distance between same token embeddings across subblocks

        /!\ This function assumes a specifc folder architecture (see savefig function)
        /!\ This function assumes that activations_z has been populated

        Args:
        - board_abs: (h,w) numpy array of abstraction indices
        - folder: folder to save plots in

        Returns:
            None
        """

        # go through activation and find min/max values for normalization across plots
        activations = {}
        act_min = np.Inf # for normalization across plots
        act_max = -np.Inf # for normalization across plots

        for k, v in self.activations_z.items():
            activations[k] = v[0].cpu().detach().numpy() # remove batch dimension
            act_min = np.minimum(act_min, np.min(activations[k]))
            act_max = np.maximum(act_max, np.max(activations[k]))
        
        #..................................
        # plot (1) zs unaltered
        # show z across blocks (row) and subblocks (columns)
        print('plot 1: z across blocks and sub-blocks')
        #..................................
        
        populations = ['z', 'attn', 'z_attn', 'mlp', 'z_attn_mlp']

        fig, axes = plt.subplots(self.n_layer, len(populations), figsize = (len(populations)*3, self.n_layer*3))

        for idx_block in range(self.n_layer):
            for idx_pop in range(len(populations)):
                
                z = activations[f'b_{idx_block}_{populations[idx_pop]}']
                axes[idx_block, idx_pop].imshow(z, cmap = 'magma')

                if idx_block == 0:
                    axes[idx_block, idx_pop].title.set_text(populations[idx_pop])
                if idx_pop == 0:
                    axes[idx_block, idx_pop].set(ylabel=f'block {idx_block}')

        plt.savefig(f'./runs/{self.run_name}/analysis/{folder}/embeddings_z.jpg', bbox_inches='tight', dpi=self.plot_dpi)
        plt.close()

        #..................................
        # plot (2) norm(z) across blocks (row) and subblocks (columns), arrange to show board structure
        print('plot 2: norm(z) across blocks and sub-blocks')
        #..................................

        fig, axes = plt.subplots(self.n_layer, len(populations), figsize = (len(populations)*3, self.n_layer*3))

        for idx_block in range(self.n_layer):
            for idx_pop in range(len(populations)):
                
                z = activations[f'b_{idx_block}_{populations[idx_pop]}']
                z_norms = np.zeros((z.shape[0],))

                for k in range(z.shape[0]):
                    z_norms[k] = np.linalg.norm(z[k])

                axes[idx_block, idx_pop].imshow(rearrange(z_norms, '(h w) -> h w', h = self.dataset.board_dim), cmap = 'magma',vmin = 0.)

                if idx_block == 0:
                    axes[idx_block, idx_pop].title.set_text(populations[idx_pop])
                if idx_pop == 0:
                    axes[idx_block, idx_pop].set(ylabel=f'block {idx_block}')

        plt.savefig(f'./runs/{self.run_name}/analysis/{folder}/embeddings_z_norm.jpg', bbox_inches='tight', dpi=self.plot_dpi)
        plt.close()

        #..................................
        # plot (3.1) & (3.2) distance between different token embeddings within subblocks
        print('plot 3: distance between different token embeddings within subblocks')
        #..................................

        fig, axes = plt.subplots(self.n_layer, len(populations), figsize = (len(populations)*3, self.n_layer*3))
        imgs = []

        for idx_block in range(self.n_layer):
            for idx_pop in range(len(populations)):
                
                z = activations[f'b_{idx_block}_{populations[idx_pop]}']
                z_plot = dist(z)
                #z_plot = dist_unit(z)

                # add abstraction information
                # get all abstractions by looking at board_abs
                abs_indices = np.unique(board_abs)
                abs_indices = abs_indices[abs_indices > -1] # remove background
                z_abs = np.linspace(0., np.max(z_plot), len(abs_indices)+1)[1:] # plot intensity for each abs

                board_abs_flat = rearrange(board_abs, 'h w -> (h w)')
                assert board_abs_flat.shape[0] == z_plot.shape[0], "dimension mismatch"

                for i in range(z_plot.shape[0]):
                    if board_abs_flat[i] > -1: # token belongs to abstraction
                        j = np.where(abs_indices == board_abs_flat[i])[0][0]
                        z_plot[i,:i+1] = z_abs[j]


                imgs.append(axes[idx_block, idx_pop].imshow(z_plot, cmap = 'magma'))
                plt.colorbar(imgs[-1], ax=axes[idx_block, idx_pop])

                if idx_block == 0:
                    axes[idx_block, idx_pop].title.set_text(populations[idx_pop])
                if idx_pop == 0:
                    axes[idx_block, idx_pop].set(ylabel=f'block {idx_block}')
                if idx_block == self.n_layer-1 and idx_pop == len(populations) - 1:
                    axes[idx_block, idx_pop].title.set_text('euclidean')

        plt.savefig(f'./runs/{self.run_name}/analysis/{folder}/embeddings_z_dist.jpg', bbox_inches='tight', dpi=self.plot_dpi)
        plt.close()

        fig, axes = plt.subplots(self.n_layer, len(populations), figsize = (len(populations)*3, self.n_layer*3))
        imgs = []

        for idx_block in range(self.n_layer):
            for idx_pop in range(len(populations)):
                
                z = activations[f'b_{idx_block}_{populations[idx_pop]}']
                z_plot = dist_unit(z)
                #z_plot = dist_unit(z)

                # add abstraction information
                # get all abstractions by looking at board_abs
                abs_indices = np.unique(board_abs)
                abs_indices = abs_indices[abs_indices > -1] # remove background
                z_abs = np.linspace(0., np.max(z_plot), len(abs_indices)+1)[1:] # plot intensity for each abs

                board_abs_flat = rearrange(board_abs, 'h w -> (h w)')
                assert board_abs_flat.shape[0] == z_plot.shape[0], "dimension mismatch"

                for i in range(z_plot.shape[0]):
                    if board_abs_flat[i] > -1: # token belongs to abstraction
                        j = np.where(abs_indices == board_abs_flat[i])[0][0]
                        z_plot[i,:i+1] = z_abs[j]


                imgs.append(axes[idx_block, idx_pop].imshow(z_plot, cmap = 'magma'))
                plt.colorbar(imgs[-1], ax=axes[idx_block, idx_pop])

                if idx_block == 0:
                    axes[idx_block, idx_pop].title.set_text(populations[idx_pop])
                if idx_pop == 0:
                    axes[idx_block, idx_pop].set(ylabel=f'block {idx_block}')
                if idx_block == self.n_layer-1 and idx_pop == len(populations) - 1:
                    axes[idx_block, idx_pop].title.set_text('euclidean unit')

        plt.savefig(f'./runs/{self.run_name}/analysis/{folder}/embeddings_z_dist_unit.jpg', bbox_inches='tight', dpi=self.plot_dpi)
        plt.close()
        
        #..................................
        # plot (4) distances between same z across subblocks
        print('plot 4: distances between same z across sub-blocks')
        #..................................

        # add abstraction information
        # get all abstractions by looking at board_abs
        abs_indices = np.unique(board_abs)
        abs_indices = abs_indices[abs_indices > -1] # remove background
        plot_colors = ['tab:'+ c for c in ['blue','orange','green','red','purple','brown','pink','olive']]

        n, d = activations[f'b_{0}_{populations[0]}'].shape
        Z = np.zeros((self.n_layer * len(populations), n, d))
        board_abs_flat = rearrange(board_abs, 'h w -> (h w)').astype(int)

        for idx_block in range(self.n_layer):
            for idx_pop in range(len(populations)):
                i = idx_block * len(populations) + idx_pop
                Z[i,:,:] = activations[f'b_{idx_block}_{populations[idx_pop]}']
        
        # make a square plot m*m ~ n 
        m1, m2 = find_best_plot_dim(n)

        fig, axes = plt.subplots(m1, m2, figsize = (m2*2, m1*2))
        imgs = []

        # find min and max
        v_min = np.Inf # for normalization across plots
        v_max = -np.Inf # for normalization across plots
        for idx_plot in range(n):
            z_plot = dist_unit(Z[:,idx_plot,:])
            v_min = np.minimum(v_min, np.min(z_plot))
            v_max = np.maximum(v_max, np.max(z_plot))


        for idx_plot in range(n):
            
            i = idx_plot // m1
            j = idx_plot - i * m1

            z_plot = dist_unit(Z[:,idx_plot,:]) # (l,d) -> (l,l)
            
            imgs.append(axes[i, j].imshow(z_plot, cmap = 'magma', vmin = v_min, vmax = v_max))

            for idx_block in range(self.n_layer):
                axes[i, j].axhline(y = idx_block * len(populations)-0.5, color = 'w', linestyle = '-', linewidth = 0.5)
                axes[i, j].axvline(x = idx_block * len(populations)-0.5, color = 'w', linestyle = '-', linewidth = 0.5)

            plt.colorbar(imgs[-1], ax=axes[i, j])
            if board_abs_flat[idx_plot] > -1:
                axes[i, j].title.set_text(f't-{idx_plot}*{board_abs_flat[idx_plot]}')
                k = Z.shape[0] // 5
                l = np.where(abs_indices == board_abs_flat[idx_plot])[0][0]
                rect = Rectangle((-0.5,Z.shape[0]-k-0.5),k,k,linewidth=2,edgecolor='none',facecolor= plot_colors[l])
                axes[i,j].add_patch(rect)
            else:
                axes[i, j].title.set_text(f't-{idx_plot}')
            axes[i,j].get_xaxis().set_visible(False)
            axes[i,j].get_yaxis().set_visible(False)

        plt.savefig(f'./runs/{self.run_name}/analysis/{folder}/embeddings_z_dist_across.jpg', bbox_inches='tight', dpi=self.plot_dpi)
        plt.close()

    # sub-function to `analyze_all`
    def analyze_all_z_2D(self, board_abs, folder = 'default'):

        """
        For each block and sublblock, shows distance between each token embedding and the others.

        /!\ This function assumes a specifc folder architecture (see savefig function)
        /!\ This function assumes that activations_z has been populated

        Args:
        - board_abs: (h,w) numpy array of abstraction indices
        - folder: folder to save plots in

        Returns:
            None
        """

        # go through activation and find min/max values for normalization across plots
        activations = {}
        act_min = np.Inf # for normalization across plots
        act_max = -np.Inf # for normalization across plots

        for k, v in self.activations_z.items():
            activations[k] = v[0].cpu().detach().numpy() # remove batch dimension
            act_min = np.minimum(act_min, np.min(activations[k]))
            act_max = np.maximum(act_max, np.max(activations[k]))

        populations = ['z', 'attn', 'z_attn', 'mlp', 'z_attn_mlp']
        
        fig, axes = plt.subplots(self.n_layer, len(populations), figsize = (len(populations)*3, self.n_layer*3), dpi = 300)
        imgs = []

        for idx_block in range(self.n_layer):
            for idx_pop in range(len(populations)):
                
                z = activations[f'b_{idx_block}_{populations[idx_pop]}']
                D = dist_unit(z)
                D = D + D.T 
                # remove diagonal
                D = D - np.diag(np.diag(D))

                D = rearrange(D, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', 
                      h1 = self.dataset.board_dim,
                      h2 = self.dataset.board_dim)
                D = rearrange(D, 'h1 w1 h2 w2 -> (h1 h2) (w1 w2)')

                # plot D
                imgs.append(axes[idx_block, idx_pop].imshow(D, cmap = 'magma'))
                
                # add horizontal and vertical lines every self.dataset.board_dim
                for k in range(1,self.dataset.board_dim):
                    axes[idx_block, idx_pop].axhline(y = k * self.dataset.board_dim-0.5, color = 'w', linestyle = '-', linewidth = 0.5)
                    axes[idx_block, idx_pop].axvline(x = k * self.dataset.board_dim-0.5, color = 'w', linestyle = '-', linewidth = 0.5)

                # add scatter of points to mark query token
                queries_bck = []
                queries_obj = []
                for i in range(self.dataset.board_dim):
                    for j in range(self.dataset.board_dim):
                        x = i * self.dataset.board_dim + i
                        y = j * self.dataset.board_dim + j
                        if board_abs[i,j] == -1:
                            queries_bck.append(np.array([x, y]).reshape((1,2)))
                        else:
                            queries_obj.append(np.array([x, y]).reshape((1,2)))

                if len(queries_bck) > 0:
                    queries_bck = np.concatenate(queries_bck, axis = 0)
                    axes[idx_block, idx_pop].scatter(queries_bck[:,0], queries_bck[:,1], s = 2, c = 'cyan', marker = 's')
                if len(queries_obj) > 0:
                    queries_obj = np.concatenate(queries_obj, axis = 0)
                    axes[idx_block, idx_pop].scatter(queries_obj[:,0], queries_obj[:,1], s = 2, c = 'magenta', marker = 's')

                # remove axis
                axes[idx_block, idx_pop].get_xaxis().set_visible(False)
                axes[idx_block, idx_pop].get_yaxis().set_visible(False)
                # add title
                if idx_block == 0:
                    axes[idx_block, idx_pop].title.set_text(populations[idx_pop])

        plt.savefig(f'./runs/{self.run_name}/analysis/{folder}/embeddings_z_2D.jpg', bbox_inches='tight', dpi=self.plot_dpi)
        plt.close()

    # sub-function to `analyze_all`
    def analyze_all_attn(self, board, board_abs, masked_board, board_recons, queries, folder = 'default'):

        """
        create plots to visualize the attention masks throughout all computational stages
        
        /!\ This function assumes a specifc folder architecture (see savefig function)
        /!\ This function assumes that activations_attn has been populated

        Args:
        - board: (h,w) numpy array of tokens
        - board_abs: (h,w) numpy array of abstraction indices
        - masked_board: (h,w) numpy array of tokens with masking
        - board_recons: (h,w) numpy array of tokens with masking
        - queries: (h,w) numpy array indicating the query token for attention maps. -1 if not a query token, otherwise the index of the query token
        - folder: folder to save plots in

        Returns:
            None
        """

        n_q = int(np.sum(queries != -1)) # number of queries
        plot_dim_i, plot_dim_j = find_best_plot_dim(n_q)
        plot_dim_i = max(2, plot_dim_i)
        plot_dim_j = max(2, plot_dim_j)

        print(f'{n_q} -> {plot_dim_i}x{plot_dim_j}')

        h, w = board.shape

        for idx_head in range(-1, self.n_head): #cycle over attn heads

            # convert to numpy for export
            activations = {}
            for k, v in self.activations_attn.items():
                if idx_head == -1:
                    activations[k] = v[0].cpu().detach().numpy()
                    #average across attention heads
                    activations[k] = reduce(activations[k], 'h t1 t2 -> t1 t2', 'mean')
                else:
                    activations[k] = v[0,idx_head,:,:].cpu().detach().numpy() # grab one head at a time
                # reshape to get attention map (h,h) for each token query (t)
                activations[k] = rearrange(activations[k], 't (h w) -> t h w', h = h) # (b,t,h,h)
                activations[k] = rearrange(activations[k], '(t1 t2) h w -> t1 t2 h w', t1 = h) # (b,t,t,h,h)

            for idx_block in range(self.n_layer): # cycle over blocks

                curr_attn = activations[f'attn_{idx_block}']

                fig, axes = plt.subplots(plot_dim_i, plot_dim_j, figsize = (plot_dim_j * 3, plot_dim_i*3))
            
                for idx_q in range(n_q):

                    plot_i = idx_q // plot_dim_j
                    plot_j = idx_q - plot_i * plot_dim_j
                    ci, cj = get_ij(queries, idx_q)
                    #print(f'plotting {plot_i},{plot_j} for query {ci},{cj}')

                    axes[plot_i,plot_j].imshow(board, cmap = 'Set3', vmin=0, vmax=self.vocab_size)

                    # add labels
                    for k in range(h):
                        for l in range(h):
                            if board_abs[k,l] >= 0: #if there is an abstraction there
                                axes[plot_i,plot_j].text(l-0.2, k+0.2, str(board[k,l]), size=8, color = 'darkslategray')            

                    # draw baord grid lines
                    for j in range(1,h):
                        rect = Rectangle((j-0.5,-0.5),1,h,linewidth=1,edgecolor='gainsboro',facecolor='none')
                        axes[plot_i,plot_j].add_patch(rect)

                        rect = Rectangle((-0.5,j-0.5),h,1,linewidth=1,edgecolor='gainsboro',facecolor='none')
                        axes[plot_i,plot_j].add_patch(rect)

                    # remove ticks
                    axes[plot_i,plot_j].tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)

                    # circle masked & add attn
                    for k in range(h):
                        for l in range(h):
                            if masked_board[k,l] == self.masking_token: # token to predict
                                rect = Rectangle((l-0.5,k-0.5),1,1,linewidth=2,edgecolor='k',facecolor='none')
                                axes[plot_i,plot_j].add_patch(rect)
                    
                    # mark query token
                    rect = Rectangle((cj-0.5,ci-0.5),1,1,linewidth=2,edgecolor='w',facecolor='none')
                    axes[plot_i,plot_j].add_patch(rect)
                    axes[plot_i,plot_j].text(cj-0.2, ci+0.2, 'x', size=8, color = 'w')

                    # (to each recon) add attn for that token
                    attn_max = np.max(curr_attn[ci,cj])
                    for k in range(h):
                        for l in range(h):
                            alpha = 0.75 * curr_attn[ci,cj,k,l]/attn_max
                            rect = Rectangle((l - 0.5,k - 0.5),alpha,alpha,linewidth=1,edgecolor='none',facecolor='darkslategray', alpha = 0.50)
                            axes[plot_i,plot_j].add_patch(rect)
                    
                    # circle mistakes
                    for k in range(h):
                        for l in range(h):
                            if board[k,l] != board_recons[k,l]:
                                rect = Rectangle((l-0.5,k-0.5),1,1,linewidth=2,edgecolor='r',facecolor='none')
                                axes[plot_i,plot_j].add_patch(rect)

                plt.xlabel(f'b{idx_block}_h{idx_head}')
                filename = f'./runs/{self.run_name}/analysis/{folder}/attn_b{idx_block}_h{idx_head}.jpg'
                print(f'...... exporting testing @ {filename}')
                plt.savefig(filename, bbox_inches='tight', dpi=self.plot_dpi)
                plt.close()

        merge_attn_plots(f'./runs/{self.run_name}/analysis/{folder}/', 
                         [i for i in range(self.n_head)], 
                         [i for i in range(self.n_layer)])

    # sub-function to `analyze_all`
    def analyze_all_attn_simple(self, board_abs, folder = 'default'):

        """
        create plots to visualize the attention masks throughout all computational stages. Each token is in turn cosidered a query`.
        
        /!\ This function assumes a specifc folder architecture (see savefig function)
        /!\ This function assumes that activations_attn has been populated

        Args:
        - board_abs: (h,w) numpy array of abstraction indices
        - folder: folder to save plots in

        Returns:
            None
        """

        # get board width and height
        h = self.dataset.board_dim

        # make a plot with n_head rows and nb_layers columns
        fig, axes = plt.subplots(self.n_head, self.n_layer, figsize = (self.n_layer*3, self.n_head*3))

        # activations_attn dictionary has just been filled
        for idx_block in range(self.n_layer): # cycle over blocks

            attn = self.activations_attn[f'attn_{idx_block}']
                
            for idx_head in range(self.n_head): #cycle over attn heads

                attn_head = attn[0,idx_head,:,:].cpu().detach().numpy() # grab one head at a time
                
                # normalize to bring the highest value in each row to 1
                print(f'{idx_block},{idx_head} -> [{np.min(attn_head)},{np.max(attn_head)}]')
                attn_head = attn_head / np.max(attn_head, axis = 1, keepdims = True)
                attn_head = np.log(attn_head + 1e-10) # log to make small values more visible
                # reshape to get attention map (h,h) for each token query (t)
                attn_head = rearrange(attn_head, 't (h w) -> t h w', h = h) # (b,t,h,h)
                attn_head = rearrange(attn_head, '(t1 t2) h w -> (t1 h) (t2 w)', t1 = h) # (b,t,t,h,h)

                # plot activations[k]
                axes[idx_head, idx_block].imshow(attn_head, cmap = 'viridis')

                # add horizontal and vertical lines every self.dataset.board_dim
                for k in range(1,h):
                    axes[idx_head, idx_block].axhline(y = k * h-0.5, color = 'w', linestyle = '-', linewidth = 0.5)
                    axes[idx_head, idx_block].axvline(x = k * h-0.5, color = 'w', linestyle = '-', linewidth = 0.5)

                # remove axis
                axes[idx_head, idx_block].get_xaxis().set_visible(False)
                axes[idx_head, idx_block].get_yaxis().set_visible(False)

                # add scatter of points to mark query token
                queries_bck = []
                queries_obj = []
                for i in range(self.dataset.board_dim):
                    for j in range(self.dataset.board_dim):
                        x = i * self.dataset.board_dim + i
                        y = j * self.dataset.board_dim + j
                        if board_abs[i,j] == -1:
                            queries_bck.append(np.array([x, y]).reshape((1,2)))
                        else:
                            queries_obj.append(np.array([x, y]).reshape((1,2)))

                if len(queries_bck) > 0:
                    queries_bck = np.concatenate(queries_bck, axis = 0)
                    axes[idx_head, idx_block].scatter(queries_bck[:,0], queries_bck[:,1], s = 2, c = 'red', marker = 's')
                if len(queries_obj) > 0:
                    queries_obj = np.concatenate(queries_obj, axis = 0)
                    axes[idx_head, idx_block].scatter(queries_obj[:,0], queries_obj[:,1], s = 2, c = 'magenta', marker = 's')

        # use tight layout
        plt.tight_layout()
        # save to disk
        img_path = f'./runs/{self.run_name}/analysis/{folder}/attn.jpg'
        print(f'...... exporting @ {img_path}')
        plt.savefig(img_path, bbox_inches='tight', dpi=self.plot_dpi)
        plt.close()

    # visualize token embeddings across all blocks and sub-blocks, attention maps across all blocks and sub-blocks    
    @torch.no_grad()
    def analyze_all(self, board, board_abs, mask, queries, folder = 'default', bool_z = True, bool_attn = True):

        """
        takes board and mask and runs it through the model, then plots z and attention maps throughout all computational stages.
        
        Args:
            board: numpy (n,n) int board
            mask: numpy binary (1 = token will be masked), same shape as board, indicates which tokens will be masked
            queries: numpy (n,n) -1 = don't investigate, i = investigat as ith, attention maps will be plotted for each query
            folder: str, folder to save plots to
            bool_z: bool, whether to plot z embeddings
            bool_attn: bool, whether to plot attention maps

        Returns:
            None
        """

        h, w = board.shape

        #..................................
        # mask board
        #..................................
        board = torch.from_numpy(rearrange(board,'h w -> 1 (h w)')).long().to(self.device) # convert to tensor (1,n*n)
        mask = torch.from_numpy(rearrange(mask,'h w -> 1 (h w)')).long().to(self.device) # convert to tensor (1,n*n)

        fully_masked_board = self.masking_token * torch.ones(board.shape, device = self.device)
        masked_board = (mask * fully_masked_board + (1-mask) * board).long() # should be indices

        #..................................
        # register hooks & get activations
        #..................................
        # prepare hooks
        self._register_hooks_attn()
        self.activations_attn = {} # empty dict
        self._register_hooks_z()
        self.activations_z = {} # empty dict

        logits = self.transformer(masked_board) # (b,n*n,v)
        board_recons = torch.argmax(F.softmax(logits, dim = -1), dim = -1)
        self._unregister_hooks() # unregister hooks

        #..................................
        # convert all to numpy
        #..................................
        board = rearrange(board[0], '(h w) -> h w', h = h).cpu().detach().numpy()
        masked_board = rearrange(masked_board[0], '(h w) -> h w', h = h).cpu().detach().numpy()
        board_recons = rearrange(board_recons[0], '(h w) -> h w', h = h).cpu().detach().numpy()

        print(f'board:\n{board}')
        print(f'masked_board:\n{masked_board}')
        print(f'board recons:\n{board_recons}')

        #..................................
        # z analysis
        #..................................
        if bool_z:
            self.analyze_all_z(board_abs, folder)
            self.analyze_all_z_2D(board_abs, folder)
        
        #..................................
        # attn analysis
        #..................................
        if bool_attn:
            self.analyze_all_attn(board, board_abs, masked_board, board_recons, queries, folder)
            self.analyze_all_attn_simple(board_abs, folder)

    # compare z embeddings across boards
    @torch.no_grad()
    def analyze_across_boards(self, boards, masks, focus_masks = None, folder = 'default'):

        """
        conduct a comparative analysis of the z embeddings across boards.

        /!\ This function assumes a specifc folder architecture (see savefig function)
        
        Args:
            boards: numpy (k,n,n) int k boards
            masks: numpy binary (k,n,n) 1 = token will be masked, same shape as board, indicates which tokens will be masked
            focus_masks: (k,n,n) binary 1 = token focused on to establish colormap min/max (default = None, in which case min/max are computed across all tokens)
            folder: str, folder to save plots to

        Returns:
            None
        """
        batch, h, w = boards.shape

        #..................................
        # mask board
        #..................................
        boards = torch.from_numpy(rearrange(boards,'b h w -> b (h w)')).long().to(self.device) # convert to tensor (k,n*n)
        masks = torch.from_numpy(rearrange(masks,'b h w -> b (h w)')).long().to(self.device) # convert to tensor (1,n*n)

        fully_masked_boards = self.masking_token * torch.ones(boards.shape, device = self.device)
        masked_boards = (masks * fully_masked_boards + (1-masks) * boards).long() # should be indices

        print(f'dim of masked boards: {masked_boards.size()}')
        for i in range(masked_boards.size(0)):
            print(f'board {i}:\n{masked_boards[i]}')

        #..................................
        # register hooks & get activations
        #..................................
        # prepare hooks
        self._register_hooks_attn()
        self.activations_attn = {} # empty dict
        self._register_hooks_z()
        self.activations_z = {} # empty dict

        logits = self.transformer(masked_boards) # (b,n*n,v)
        boards_recons = torch.argmax(F.softmax(logits, dim = -1), dim = -1)
        self._unregister_hooks() # unregister hooks

        #..................................
        # convert all to numpy
        #..................................
        boards = rearrange(boards, 'b (h w) -> b h w', h = h).cpu().detach().numpy()
        masked_boards = rearrange(masked_boards, 'b (h w) -> b h w', h = h).cpu().detach().numpy()
        boards_recons = rearrange(boards_recons, 'b (h w) -> b h w', h = h).cpu().detach().numpy()
        for i in range(boards_recons.shape[0]):
            print(f'recons:\n{boards_recons[i]}')

        # go through activation and find min/max values for normalization across plots
        activations = {}
        act_min = np.Inf # for normalization across plots
        act_max = -np.Inf # for normalization across plots

        for k, v in self.activations_z.items():
            activations[k] = v.cpu().detach().numpy() # remove batch dimension
            act_min = np.minimum(act_min, np.min(activations[k]))
            act_max = np.maximum(act_max, np.max(activations[k]))
        
        #..................................
        # display board
        #..................................
        for idx_k in range(batch):
            print(f'board:\n{boards[idx_k]}')
            # print(f'masked_board:\n{masked_boards[idx_k]}')
            # print(f'board recons:\n{boards_recons[idx_k]}')

        
        populations = ['z', 'attn', 'z_attn', 'mlp', 'z_attn_mlp']

        #..................................
        # compute vmin and vmax
        #..................................

        D_min = np.Inf # for normalization across plots
        D_max = -np.Inf # for normalization across plots
        
        if focus_masks is None:
            focus_masks = np.ones((batch, h, w))
        focus_masks = rearrange(focus_masks, 'k h w -> k (h w)')
        F_dim = int(np.sum(focus_masks[0]))


        for idx_block in range(self.n_layer):
            for idx_pop in range(len(populations)):

                Z = activations[f'b_{idx_block}_{populations[idx_pop]}'] # (k,t,d)
                _, nb_tokens, embd_dim = Z.shape
                Z_focus = np.zeros((batch, F_dim, embd_dim))

                for b in range(batch):
                    Z_focus[b] = Z[b][focus_masks[b] == 1]

                D = np.zeros((Z_focus.shape[1],))

                for idx_token in range(Z_focus.shape[1]):
                    z = Z_focus[:,idx_token,:]
                    d = dist_unit(z)
                    s, counter = 0., 0
                    for k in range(d.shape[0]-1):
                        for l in range(k+1,d.shape[0]):
                            s += d[k,l]
                            counter += 1

                    D[idx_token] = s / counter

                D_min = np.minimum(D_min, np.min(D))
                D_max = np.maximum(D_max, np.max(D))

        print(f'vmin = {D_min},, vmax = {D_max}')
        
        #..................................
        # plot
        #..................................

        fig, axes = plt.subplots(self.n_layer, len(populations), figsize = (len(populations)*3, self.n_layer*3))
        imgs = []

        for idx_block in range(self.n_layer):
            for idx_pop in range(len(populations)):

                Z = activations[f'b_{idx_block}_{populations[idx_pop]}'] # (k,t,d)
                _, nb_tokens, embd_dim = Z.shape
                Z_focus = np.zeros((batch, F_dim, embd_dim))

                for b in range(batch):
                    Z_focus[b] = Z[b][focus_masks[b] == 1]

                D = np.zeros((Z_focus.shape[1],))

                for idx_token in range(Z_focus.shape[1]):
                    z = Z_focus[:,idx_token,:]
                    d = dist_unit(z)
                    s, counter = 0., 0
                    for k in range(d.shape[0]-1):
                        for l in range(k+1,d.shape[0]):
                            s += d[k,l]
                            counter += 1

                    D[idx_token] = s / counter

                D = rearrange(D, '(h w) -> h w', h = int(np.sqrt(F_dim)))

                imgs.append(axes[idx_block, idx_pop].imshow(D, cmap = 'magma',vmin = D_min, vmax = D_max))
                plt.colorbar(imgs[-1], ax=axes[idx_block, idx_pop])

                if idx_block == 0:
                    axes[idx_block, idx_pop].title.set_text(populations[idx_pop])
                if idx_pop == 0:
                    axes[idx_block, idx_pop].set(ylabel=f'block {idx_block}')

                if idx_block == self.n_layer - 1 and  idx_pop == len(populations)-1:
                    axes[idx_block, idx_pop].title.set_text('euclidean unit')

        plt.savefig(f'./runs/{self.run_name}/analysis/{folder}/embeddings_z_acrossBoards.jpg', bbox_inches='tight', dpi=self.plot_dpi)
        plt.close()
    
    # compare attention maps across boards
    @torch.no_grad()
    def analyze_across_boards_attn(self, boards, masks, queries, folder = 'default'):

        """
        conduct a comparative analysis of the attention maps across boards
        
        /!\ This function assumes a specifc folder architecture (see savefig function)

        Args:
            boards: numpy (k,n,n) int k boards
            masks: numpy binary (k,n,n) 1 = token will be masked, same shape as board, indicates which tokens will be masked
            queries: numpy (k,n,n) -1 = don't investigate, i = investigat as ith, attention maps will be plotted for each query
            folder: str, folder to save plots to

        Returns:
            None
        """
        batch, h, w = boards.shape

        #..................................
        # mask board
        #..................................
        boards = torch.from_numpy(rearrange(boards,'b h w -> b (h w)')).long().to(self.device) # convert to tensor (k,n*n)
        masks = torch.from_numpy(rearrange(masks,'b h w -> b (h w)')).long().to(self.device) # convert to tensor (1,n*n)

        fully_masked_boards = self.masking_token * torch.ones(boards.shape, device = self.device)
        masked_boards = (masks * fully_masked_boards + (1-masks) * boards).long() # should be indices

        #..................................
        # register hooks & get activations
        #..................................
        # prepare hooks
        self._register_hooks_attn()
        self.activations_attn = {}
        self._register_hooks_z()
        self.activations_z = {}

        logits = self.transformer(masked_boards) # (b,n*n,v)
        boards_recons = torch.argmax(F.softmax(logits, dim = -1), dim = -1)
        self._unregister_hooks() # unregister hooks

        #..................................
        # convert all to numpy
        #..................................
        boards = rearrange(boards, 'b (h w) -> b h w', h = h).cpu().detach().numpy()
        masked_boards = rearrange(masked_boards, 'b (h w) -> b h w', h = h).cpu().detach().numpy()
        boards_recons = rearrange(boards_recons, 'b (h w) -> b h w', h = h).cpu().detach().numpy()
        
        #..................................
        # display board
        #..................................
        for idx_k in range(batch):
            print(f'board:\n{boards[idx_k]}')
            # print(f'masked_board:\n{masked_boards[idx_k]}')
            # print(f'board recons:\n{boards_recons[idx_k]}')

        #//////////////////////////////////
        # attn plot
        #//////////////////////////////////

        n_q = int(np.sum(queries != -1)) # number of queries
        plot_dim_i, plot_dim_j = find_best_plot_dim(n_q)
        plot_dim_i = max(2, plot_dim_i)
        plot_dim_j = max(2, plot_dim_j)

        print(f'{n_q} -> {plot_dim_i}x{plot_dim_j}')

        batch, h, w = boards.shape

        for idx_head in range(self.n_head): #cycle over attn heads

            # convert to numpy for export
            activations = {}
            for k, v in self.activations_attn.items():
                attn = v.cpu().detach().numpy() # grab one head at a time
                attn = np.var(attn, axis = 0)
                activations[k] = attn[idx_head,:,:]
                # reshape to get attention map (h,h) for each token query (t)
                activations[k] = rearrange(activations[k], 't (h w) -> t h w', h = h) # (b,t,h,h)
                activations[k] = rearrange(activations[k], '(t1 t2) h w -> t1 t2 h w', t1 = h) # (b,t,t,h,h)

            for idx_block in range(self.n_layer): # cycle over blocks

                curr_attn = activations[f'attn_{idx_block}']

                fig, axes = plt.subplots(plot_dim_i, plot_dim_j, figsize = (plot_dim_j * 3, plot_dim_i*3))
            
                for idx_q in range(n_q):

                    plot_i = idx_q // plot_dim_j
                    plot_j = idx_q - plot_i * plot_dim_j
                    ci, cj = get_ij(queries, idx_q)
                    #print(f'plotting {plot_i},{plot_j} for query {ci},{cj}')

                    axes[plot_i,plot_j].imshow(boards[0], cmap = 'Set3', vmin=0, vmax=self.vocab_size)

                    # # add labels
                    # for k in range(h):
                    #     for l in range(h):
                    #         if board_abs[k,l] >= 0: #if there is an abstraction there
                    #             axes[plot_i,plot_j].text(l-0.2, k+0.2, str(boards[0,k,l]), size=8, color = 'darkslategray')            

                    # draw baord grid lines
                    for j in range(1,h):
                        rect = Rectangle((j-0.5,-0.5),1,h,linewidth=1,edgecolor='gainsboro',facecolor='none')
                        axes[plot_i,plot_j].add_patch(rect)

                        rect = Rectangle((-0.5,j-0.5),h,1,linewidth=1,edgecolor='gainsboro',facecolor='none')
                        axes[plot_i,plot_j].add_patch(rect)

                    # remove ticks
                    axes[plot_i,plot_j].tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)

                    # # circle masked & add attn
                    # for k in range(h):
                    #     for l in range(h):
                    #         if masked_board[k,l] == self.masking_token: # token to predict
                    #             rect = Rectangle((l-0.5,k-0.5),1,1,linewidth=2,edgecolor='k',facecolor='none')
                    #             axes[plot_i,plot_j].add_patch(rect)
                    
                    # mark query token
                    rect = Rectangle((cj-0.5,ci-0.5),1,1,linewidth=2,edgecolor='w',facecolor='none')
                    axes[plot_i,plot_j].add_patch(rect)
                    axes[plot_i,plot_j].text(cj-0.2, ci+0.2, 'x', size=8, color = 'w')

                    # (to each recon) add attn for that token
                    attn_max = np.max(curr_attn[ci,cj])
                    for k in range(h):
                        for l in range(h):
                            alpha = 0.75 * curr_attn[ci,cj,k,l]/attn_max
                            rect = Rectangle((l - 0.5,k - 0.5),alpha,alpha,linewidth=1,edgecolor='none',facecolor='darkslategray', alpha = 0.50)
                            axes[plot_i,plot_j].add_patch(rect)
                    
                    # # circle mistakes
                    # for k in range(h):
                    #     for l in range(h):
                    #         if board[k,l] != board_recons[k,l]:
                    #             rect = Rectangle((l-0.5,k-0.5),1,1,linewidth=2,edgecolor='r',facecolor='none')
                    #             axes[plot_i,plot_j].add_patch(rect)

                plt.xlabel(f'b{idx_block}_h{idx_head}')
                filename = f'./runs/{self.run_name}/analysis/{folder}/attnVar_b{idx_block}_h{idx_head}.jpg'
                print(f'...... exporting testing @ {filename}')
                plt.savefig(filename, bbox_inches='tight', dpi=self.plot_dpi)
                plt.close()

        merge_attn_plots(f'./runs/{self.run_name}/analysis/{folder}/', 
                         [i for i in range(self.n_head)], 
                         [i for i in range(self.n_layer)], prefix='attnVar')
        
    #..................................
    # forward passes, used to gather activations for analysis
    # these functions are used to produce and save to disk specific activations
    #..................................
    
    @torch.no_grad()
    def forward_svg(self, board, board_abs, mask, folder = './svg'):

        """
        takes board and mask and runs it through the model, then save embeddings and attention maps throughout all computational stages to disk for further analysis (svg plots).
        
        /!\ This function assumes a specifc folder architecture (see savefig function)

        Args:
            board: numpy (n,n) int board
            board_abs: numpy matrix indicating which token belongs to which abstraction (-1 = bck)
            mask: numpy binary (1 = token will be masked), same shape as board, indicates which tokens will be masked
            folder: where to export the numpy arrays
        Returns:
            None
        """

        h, w = board.shape

        #..................................
        # mask board
        #..................................
        board = torch.from_numpy(rearrange(board,'h w -> 1 (h w)')).long().to(self.device) # convert to tensor (1,n*n)
        mask = torch.from_numpy(rearrange(mask,'h w -> 1 (h w)')).long().to(self.device) # convert to tensor (1,n*n)

        fully_masked_board = self.masking_token * torch.ones(board.shape, device = self.device)
        masked_board = (mask * fully_masked_board + (1-mask) * board).long() # should be indices

        #..................................
        # register hooks & get activations
        #..................................
        # prepare hooks
        self._register_hooks_svg()
        self.activations_svg = {} # empty dict

        logits = self.transformer(masked_board) # (b,n*n,v)
        board_recons = torch.argmax(F.softmax(logits, dim = -1), dim = -1)
        self._unregister_hooks() # unregister hooks

        #..................................
        # convert all to numpy
        #..................................
        board = rearrange(board[0], '(h w) -> h w', h = h).cpu().detach().numpy()
        masked_board = rearrange(masked_board[0], '(h w) -> h w', h = h).cpu().detach().numpy()
        board_recons = rearrange(board_recons[0], '(h w) -> h w', h = h).cpu().detach().numpy()

        print(f'board:\n{board}')
        print(f'masked_board:\n{masked_board}')
        print(f'board recons:\n{board_recons}') 

        #..................................
        # export activations to folder 
        #..................................

        meta = {
            'nb_blocks': self.n_layer,
            'nb_heads':self.n_head,
            'nb_tokens':board.size,
            'dim_emb':self.n_embd,
        }

        with open(f'{folder}/meta.json', 'w') as file:
            json.dump(meta, file)

        np.save(f'{folder}/svg_groups', board_abs.reshape((-1,)))
        np.save(f'{folder}/svg_mask', mask[0].cpu().detach().numpy().reshape((-1,)))

        for k, v in self.activations_svg.items():
            print(k, v.size())
            np.save(f'{folder}/{k}', v[0].cpu().detach().numpy())

            if k in [f'svg_b{i}_v' for i in range(self.n_layer)]:
                z = rearrange(v[0].cpu().detach().numpy(),'t (h d) -> h t d', h = self.n_head)
                for idx_head in range(self.n_head):
                    np.save(f'{folder}/{k}_h{idx_head}', z[idx_head])
            
            if k in [f'svg_b{i}_a' for i in range(self.n_layer)]:
                z = v[0].cpu().detach().numpy()
                for idx_head in range(self.n_head):
                    np.save(f'{folder}/{k}_h{idx_head}', z[idx_head])

    @torch.no_grad()
    def forward_pca(self, boards, boards_abs, masks, folder = './pca', extra_meta = {}):

        """
        takes board and mask and runs it through the model, then save embeddings and attention maps throughout all computational stages to disk for further analysis (e.g., PCA).
        
        /!\ This function assumes a specifc folder architecture (see savefig function)

        Args:
            boards: numpy (k,n,n) int k boards
            boards_abs: numpy matrix indicating which token belongs to which abstraction (-1 = bck)
            masks: numpy binary (k,n,n) 1 = token will be masked, same shape as board, indicates which tokens will be masked
            folder: where to export the numpy arrays
            extra_meta: dict, extra meta data to be saved in meta.json
        Returns:
            None
        """

        batch, h, w = boards.shape

        #..................................
        # mask board
        #..................................
        boards = torch.from_numpy(rearrange(boards,'b h w -> b (h w)')).long().to(self.device) # convert to tensor (k,n*n)
        masks = torch.from_numpy(rearrange(masks,'b h w -> b (h w)')).long().to(self.device) # convert to tensor (1,n*n)

        fully_masked_boards = self.masking_token * torch.ones(boards.shape, device = self.device)
        masked_boards = (masks * fully_masked_boards + (1-masks) * boards).long() # should be indices

        #..................................
        # register hooks & get activations
        #..................................
        # prepare hooks
        self._register_hooks_pca()
        self.activations_pca = {}

        logits = self.transformer(masked_boards) # (b,n*n,v)
        boards_recons = torch.argmax(F.softmax(logits, dim = -1), dim = -1)
        self._unregister_hooks() # unregister hooks

        #..................................
        # convert all to numpy
        #..................................
        boards = rearrange(boards, 'b (h w) -> b h w', h = h).cpu().detach().numpy()
        masks = rearrange(masks, 'b (h w) -> b h w', h = h).cpu().detach().numpy()
        masked_boards = rearrange(masked_boards, 'b (h w) -> b h w', h = h).cpu().detach().numpy()
        boards_recons = rearrange(boards_recons, 'b (h w) -> b h w', h = h).cpu().detach().numpy()

        #..................................
        # compute cumulative error between original and reconstructed board
        #..................................
        error = np.sum(np.abs(boards - boards_recons) > 0)
        print(f'error across all boards = {error} (over {boards.size} tokens)')
        for idx_board in range(boards.shape[0]):
            curr_error = np.sum(np.abs(boards[idx_board] - boards_recons[idx_board]) > 0)
            if curr_error > 0:
                print('-'*50,f'\nboard {idx_board}: error = {curr_error}')
                print('\n',boards[idx_board])
                print('\n',masked_boards[idx_board])
                print('\n',boards_recons[idx_board])

        #..................................
        # export activations to folder
        #..................................

        meta = {
            'nb_blocks': self.n_layer,
            'nb_heads':self.n_head,
            'nb_tokens':self.block_size,
            'dim_emb':self.n_embd,
        }
        # add extra meta
        meta.update(extra_meta)

        with open(f'{folder}/meta.json', 'w') as file:
            json.dump(meta, file)

        np.save(f'{folder}/boards', boards)
        np.save(f'{folder}/boards_abs', boards_abs)
        np.save(f'{folder}/masked_boards', masked_boards)
        np.save(f'{folder}/masks', masks)

        for k, v in self.activations_pca.items():
            print(k, v.size())
            np.save(f'{folder}/{k}', v.cpu().detach().numpy())

            if k in [f'b{i}_v' for i in range(self.n_layer)]:
                z = rearrange(v.cpu().detach().numpy(),'b t (h d) -> h b t d', h = self.n_head)
                for idx_head in range(self.n_head):
                    np.save(f'{folder}/{k}_h{idx_head}', z[idx_head])
            
            if k in [f'b{i}_a' for i in range(self.n_layer)]:
                z = v.cpu().detach().numpy()
                z = rearrange(v.cpu().detach().numpy(),'b h t1 t2 -> h b t1 t2')
                for idx_head in range(self.n_head):
                    np.save(f'{folder}/{k}_h{idx_head}', z[idx_head])
    
    @torch.no_grad()
    def forward_pca_batch(self, boards, boards_abs, masks, folder, batch_size):

        """
        takes board and mask, split them into different batches, and runs it through the model, then save embeddings and attention maps throughout all computational stages to disk for further analysis (e.g., PCA).
        builds on forward_pca, but splits the boards into batches to avoid memory issues.

        /!\ This function assumes a specifc folder architecture (see savefig function)

        Args:
            boards: numpy (k,n,n) int k boards
            boards_abs: numpy matrix indicating which token belongs to which abstraction (-1 = bck)
            masks: numpy binary (k,n,n) 1 = token will be masked, same shape as board, indicates which tokens will be masked
            folder: where to export the numpy arrays
            batch_size: int, batch size
        """
        N, h, w = boards.shape

        nb_batches = max(1, N // batch_size)
        print(f'{N} boards, batch size = {batch_size} -> {nb_batches} batches')

        batches = divide_into_batches(list(range(N)), nb_batches)

        # sanity check
        counter = 0
        for batch in batches:
            counter += len(batch)
        assert counter == N, 'something went wrong with the batching'

        for idx_batch, batch in enumerate(batches):

            print(f'batch {idx_batch} from {batch[0]} to {batch[-1]}')
            
            #..................................
            # mask board
            #..................................
            curr_boards = torch.from_numpy(rearrange(boards[batch],'b h w -> b (h w)')).long().to(self.device) # convert to tensor (k,n*n)
            curr_masks = torch.from_numpy(rearrange(masks[batch],'b h w -> b (h w)')).long().to(self.device) # convert to tensor (1,n*n)

            fully_masked_boards = self.masking_token * torch.ones(curr_boards.shape, device = self.device)
            masked_boards = (curr_masks * fully_masked_boards + (1-curr_masks) * curr_boards).long() # should be indices

            #..................................
            # register hooks & get activations
            #..................................
            # prepare hooks
            self._register_hooks_pca()
            self.activations_pca = {}

            logits = self.transformer(masked_boards) # (b,n*n,v)
            boards_recons = torch.argmax(F.softmax(logits, dim = -1), dim = -1)
            self._unregister_hooks() # unregister hooks

            #..................................
            # convert all to numpy
            #..................................
            np_boards = rearrange(curr_boards, 'b (h w) -> b h w', h = h).cpu().detach().numpy()
            np_masks = rearrange(curr_masks, 'b (h w) -> b h w', h = h).cpu().detach().numpy()
            np_masked_boards = rearrange(masked_boards, 'b (h w) -> b h w', h = h).cpu().detach().numpy()
            np_boards_recons = rearrange(boards_recons, 'b (h w) -> b h w', h = h).cpu().detach().numpy()

            #..................................
            # export activations to folder
            #..................................

            np.save(f'{folder}/batch_{idx_batch}_boards', np_boards)
            np.save(f'{folder}/batch_{idx_batch}_boards_abs', np_masks)
            np.save(f'{folder}/batch_{idx_batch}_masked_boards', np_masked_boards)
            np.save(f'{folder}/batch_{idx_batch}_masks', np_boards_recons)

            for k, v in self.activations_pca.items():
                # print(k, v.size())
                np.save(f'{folder}/batch_{idx_batch}_{k}', v.cpu().detach().numpy())

                if k in [f'b{i}_v' for i in range(self.n_layer)]:
                    z = rearrange(v.cpu().detach().numpy(),'b t (h d) -> h b t d', h = self.n_head)
                    for idx_head in range(self.n_head):
                        np.save(f'{folder}/batch_{idx_batch}_{k}_h{idx_head}', z[idx_head])
                
                if k in [f'b{i}_a' for i in range(self.n_layer)]:
                    z = v.cpu().detach().numpy()
                    z = rearrange(v.cpu().detach().numpy(),'b h t1 t2 -> h b t1 t2')
                    for idx_head in range(self.n_head):
                        np.save(f'{folder}/batch_{idx_batch}_{k}_h{idx_head}', z[idx_head])

        meta = {
            'batches': batches,
            'nb_blocks': self.n_layer,
            'nb_heads':self.n_head,
            'nb_tokens':self.block_size,
            'dim_emb':self.n_embd,
        }

        with open(f'{folder}/meta.json', 'w') as file:
            json.dump(meta, file)
    
    @torch.no_grad()
    def forward_all_activations(self, boards, masks, edits = None, verbose = False, return_recons = False):

        """
        takes board and mask and runs it through the model, then save embeddings and attention maps throughout all computational stages to disk for further analysis (e.g., PCA).
        additional edits can be passed to the model to investigate the effect of edits on the activations.

        Args:
            boards: numpy (k,n,n) int k boards
            masks: numpy binary (k,n,n) 1 = token will be masked, same shape as board, indicates which tokens will be masked
            edits: dict, additional edits to be passed to the model (default = None, in which case no edits are passed)
            verbose: bool, if True, print some info
            return_recons: bool, if True, return the reconstructed boards
        
        Returns:
            activations_pca: dict, activations
            boards_recons: numpy (k,n,n) int k reconstructed boards (only if return_recons = True)
        """

        batch, h, w = boards.shape

        #..................................
        # mask board
        #..................................
        boards = torch.from_numpy(rearrange(boards,'b h w -> b (h w)')).long().to(self.device) # convert to tensor (k,n*n)
        masks = torch.from_numpy(rearrange(masks,'b h w -> b (h w)')).long().to(self.device) # convert to tensor (1,n*n)

        fully_masked_boards = self.masking_token * torch.ones(boards.shape, device = self.device)
        masked_boards = (masks * fully_masked_boards + (1-masks) * boards).long() # should be indices

        #..................................
        # convert edits to tensor
        #..................................
        def to_tensor_recursive(d, prefix=''):
            '''Recursively explore a dictionary'''
            for key, value in d.items():
                if isinstance(value, dict):
                    to_tensor_recursive(value, prefix=prefix+str(key)+'.')
                else:
                    #print(f'converting {prefix+str(key)} to tensor')
                    d[key] = torch.from_numpy(value).double().to(self.device)
        
        if edits is not None:
            to_tensor_recursive(edits)

        #..................................
        # register hooks & get activations
        #..................................
        # prepare hooks
        self._register_hooks_pca()
        self.activations_pca = {}
        logits = self.transformer(masked_boards, activations= edits, verbose= verbose) # (b,n*n,v)
            
        boards_recons = torch.argmax(F.softmax(logits, dim = -1), dim = -1)
        self._unregister_hooks() # unregister hooks

        #..................................
        # convert all to numpy
        #..................................
        boards = rearrange(boards, 'b (h w) -> b h w', h = h).cpu().detach().numpy()
        masks = rearrange(masks, 'b (h w) -> b h w', h = h).cpu().detach().numpy()
        masked_boards = rearrange(masked_boards, 'b (h w) -> b h w', h = h).cpu().detach().numpy()
        boards_recons = rearrange(boards_recons, 'b (h w) -> b h w', h = h).cpu().detach().numpy()

        for k, v in self.activations_pca.items():
            self.activations_pca[k] = v.cpu().detach().numpy()

        #..................................
        # compute cumulative error between original and reconstructed board
        #..................................
        error = np.sum(np.abs(boards - boards_recons) > 0)
        # print(f'error across all boards = {error} (over {boards.size} tokens)')
        # for idx_board in range(boards.shape[0]):
        #     curr_error = np.sum(np.abs(boards[idx_board] - boards_recons[idx_board]) > 0)
        #     if curr_error > 0:
        #         print('-'*50,f'\nboard {idx_board}: error = {curr_error}')
        #         print('\n',boards[idx_board])
        #         print('\n',masked_boards[idx_board])
        #         print('\n',boards_recons[idx_board])

        if return_recons:
            return self.activations_pca, boards_recons
        else:
            return self.activations_pca
#-------------------------------------------------