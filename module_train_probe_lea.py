"""
Implements the `TrainTransformer` class, a wraper to train LEA.
"""

#----------------------------------------------------
# import libraries
#----------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from einops import rearrange, reduce, repeat
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from module_transformer_lea import LEA, LEAFactor
from utils import find_best_plot_dim, get_ij, merge_attn_plots, divide_into_batches
import time
import wandb
import json
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

    """wraper to train LEA"""
    
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
                - 'vocab_speech_size' (int): size of the speech vocab
                - 'block_speech_size' (int): length of sentence
                - 'bool_blank_speech' (bool): whether to use blank speech or not
                - 'vq_type' (str): type of vector quantization, 'vanilla' or 'EMA'
                - 'loss_rfs_epoch_start' (int): epoch at which loss for reconstruction from speech starts
                - 'loss_rfs_epoch_end' (int): epoch at which loss for reconstruction from speech ends
                - 'loss_rfs_val_start' (float): scaling factor for loss for reconstruction from speech at start
                - 'loss_rfs_val_end' (float): scaling factor for loss for reconstruction from speech at end
                - 'loss_speech_epoch_start' (int): epoch at which loss for speech sparsity starts
                - 'loss_speech_epoch_end' (int): epoch at which loss for speech sparsity ends
                - 'loss_speech_val_start' (float): scaling factor for loss for speech sparsity at start
                - 'loss_speech_val_end' (float): scaling factor for loss for speech sparsity at end
        Returns:
            None
        """
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self.dataset = dataset

        if self.transformer_type == 'factor':
            print('Using LEA (Factor)')
            self.transformer = LEAFactor(vocab_size = self.vocab_size, 
                                block_size = self.block_size, 
                                n_layer = self.n_layer, 
                                n_head = self.n_head, 
                                n_embd = self.n_embd,
                                n_embd_pos = self.n_embd_pos, # this is what differs with GPTSpeaker
                                embd_pdrop = self.embd_pdrop, 
                                resid_pdrop = self.resid_pdrop, 
                                attn_pdrop = self.attn_pdrop, 
                                n_unmasked = self.n_unmasked,
                                vocab_speech_size = self.vocab_speech_size, #NEW
                                block_speech_size = self.block_speech_size,
                                vq_type = self.vq_type,
                                bool_blank_speech = self.bool_blank_speech).double().to(self.device)
        elif self.transformer_type == 'vanilla':
            # depreciated
            print('Using LEA (Vanilla) -- depreciated')
            self.transformer = LEA(vocab_size = self.vocab_size, 
                                block_size = self.block_size, 
                                n_layer = self.n_layer, 
                                n_head = self.n_head, 
                                n_embd = self.n_embd,
                                embd_pdrop = self.embd_pdrop, 
                                resid_pdrop = self.resid_pdrop, 
                                attn_pdrop = self.attn_pdrop, 
                                n_unmasked = self.n_unmasked,
                                vocab_speech_size = self.vocab_speech_size, #NEW
                                block_speech_size = self.block_speech_size,
                                vq_type = self.vq_type,
                                bool_blank_speech = self.bool_blank_speech).double().to(self.device)
        else:
            raise ValueError('transformer_type must be "factor" or "vanilla"')
        
        # setup optimizer
        self.optim = self._configure_optimizers()
        # add other attributes useful for training
        self.step = 0
        self.epoch = 0

        # create hooks to recover activations and corresponding dictionaries that will store them
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
    #.......................................
    
    # need update
    def _register_hooks_pca(self):
        
        """add hooks to recover activations throughout the transformer, for pca analysis
        - modified to work with GPTSpeakerFactor
        """

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

            self.list_hooks_pca.append(self.transformer.ps_blocks[i].selfAttn.register_forward_hook(getActivation(f'b{i}_selfAttn')))
            self.list_hooks_pca.append(self.transformer.ps_blocks[i].crossAttn.register_forward_hook(getActivation(f'b{i}_crossAttn')))
            self.list_hooks_pca.append(self.transformer.ps_blocks[i].mlp.register_forward_hook(getActivation(f'b{i}_mlp')))

            self.list_hooks_pca.append(self.transformer.ps_blocks[i].register_forward_hook(getActivation(f'b{i}_out')))
        
        # add hook on the output of the ln_f
        self.list_hooks_pca.append(self.transformer.ps_ln_out.register_forward_hook(getActivation(f'z_out')))
        self.list_hooks_pca.append(self.transformer.ps_head.register_forward_hook(getActivation(f'logit')))

    # need update
    def _unregister_hooks(self):

        """remove all hooks from the model"""

        if hasattr(self, "list_hooks_pca"):
            for h in self.list_hooks_pca:
                h.remove()
            self.list_hooks_pca = []
    
    #.......................................
    # optimizer
    #.......................................
    
    def _configure_optimizers(self):

        """configure the optimizer"""

        decay, no_decay = set(), set()
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

        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer
    
    #.......................................
    # board masking strategies
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

        stats_boardType = np.zeros((board.shape[0],))
        
        if patch_masking:

            # iterate over batch
            for i in range(board.shape[0]):

                if do_patch_masking[i] == 1: # do patch masking

                    # board will be treated differently whether it features a composite abstraction or not

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
                        #print(f'coordinates: {i_s, i_e, j_s, j_e}')
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
    
    # supports wandb
    def train(self):
        
        """
        train the transformer. Supports wandb.
        
        Args:
            None
        Returns:
            cum_losses (np.ndarray): vector of cummulative losses over training
        """
            
        # make dataloader
        self.dataset.set_type = 'train'
        dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)

        # print something about training
        print(f'training transformer ({self.transformer_type}) @ {self.run_name}')
        print(f'\n* Architecture:')
        print(f'... {self.n_embd}-D embeddings')
        if self.transformer_type == 'factor':
            print(f'... {self.n_embd_pos}-D positional embeddings')
        print(f'... {self.n_layer} layers')
        print(f'... {self.n_head} attn heads, each {self.n_embd // self.n_head} dim')
        print(f'\n* Training:')
        print(f'... {self.epochs} epochs, at {len(dataloader)} steps/epoch')
        
        # init counters, etc.
        self.step = 0
        self.epoch = 0
        start_time = time.time()
        cum_losses = []
                
        for epoch in range(self.epochs):
            
            timer_epoch = time.time()

            percent_masked = [] # to store percent of masked tokens in each batch
            # LEA specific
            losses_rfb = []
            losses_rfs = []
            losses_vq = []
            losses_speech = []
            losses_total = []
            # errors
            percent_error_rfb = []
            percent_error_rfs = []
            # usage
            vq_usage = []

            # change dataset parameters to implement curriculum
            timer_curriculum = time.time()
            if self.board_curriculum:
                self.dataset.board_types_proba = ramp(epoch, 
                                                      np.array(self.board_proba_start),
                                                      np.array(self.board_proba_end),
                                                      self.board_proba_epoch_start,
                                                      self.board_proba_epoch_end)
                dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)
            timer_curriculum = time.time() - timer_curriculum

            timer_forward = 0.
            timer_backward = 0.
            
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
                timer_forward_start = time.time()
                ps_logits, ps2_logits, vq_loss, vq_encodings = self.transformer(masked_boards)
                timer_forward += time.time() - timer_forward_start

                #....................................................
                # compute losses
                #....................................................
                # RECONSTRUCTION FROM BOARD
                loss_reconsFromBoard = F.cross_entropy(rearrange(ps_logits, 'b l v -> (b l) v'), rearrange(boards, 'b l -> (b l)'))
                
                # RECONSTRUCTION FROM SPEECH
                loss_reconsFromSpeech = F.cross_entropy(rearrange(ps2_logits, 'b l v -> (b l) v'), rearrange(boards, 'b l -> (b l)'))
                # compute mixing factor for reconstruction from speech
                # starts at zero and increase until maxing out
                # to 1. for epoch >= loss_rfs_epoch 
                alpha_rfs = ramp(epoch, self.loss_rfs_val_start, self.loss_rfs_val_end, self.loss_rfs_epoch_start, self.loss_rfs_epoch_end)  

                # LANGUAGE SPARSITY
                # encourage more usage of last token
                loss_speech =  1. - torch.mean(vq_encodings[:,:,-1])
                alpha_speech = ramp(epoch, self.loss_speech_val_start, self.loss_speech_val_end, self.loss_speech_epoch_start, self.loss_speech_epoch_end)

                # put it all together
                loss = loss_reconsFromBoard + alpha_rfs * (vq_loss + loss_reconsFromSpeech) + alpha_speech * loss_speech
                
                # get gradient and update weights
                timer_backward_start = time.time()
                loss.backward()
                self.optim.step()
                timer_backward += time.time() - timer_backward_start

                #....................................................
                # log losses and others
                #....................................................

                losses_rfb.append(loss_reconsFromBoard.cpu().detach().item())
                losses_rfs.append(loss_reconsFromSpeech.cpu().detach().item())
                losses_vq.append(vq_loss.cpu().detach().item())
                losses_speech.append(loss_speech.cpu().detach().item())
                losses_total.append(loss.cpu().detach().item())

                percent_masked.append(1. - (torch.sum(boards == masked_boards)/torch.numel(boards)).cpu().detach().item())
                
                # compute reconstruction errors
                reconstructed_boards = torch.argmax(ps_logits, dim = -1)
                error = torch.mean((reconstructed_boards != boards).float())
                percent_error_rfb.append(error.cpu().detach().item())

                reconstructed_boards2 = torch.argmax(ps2_logits, dim = -1)
                error2 = torch.mean((reconstructed_boards2 != boards).float())
                percent_error_rfs.append(error2.cpu().detach().item())

                # compute vq usage
                vq_encodings = rearrange(vq_encodings, 'b l d -> 1 (b l) d')
                vq_usage.append(torch.sum(vq_encodings, dim = 1).cpu().detach())

                #....................................................
                # end step
                #....................................................

                self.step += 1
                self.epoch = epoch
            
            cum_loss_rfb = np.sum(np.array(losses_rfb))
            cum_loss_rfs = np.sum(np.array(losses_rfs))
            cum_loss_vq = np.sum(np.array(losses_vq))
            cum_loss_speech = np.sum(np.array(losses_speech))
            cum_loss_total = np.sum(np.array(losses_total))
            cum_losses.append(cum_loss_total)
            avg_error_rfb = np.mean(np.array(percent_error_rfb))
            avg_error_rfs = np.mean(np.array(percent_error_rfs))

            vq_usage = torch.cat(vq_usage, dim = 0)
            vq_usage = torch.sum(vq_usage, dim = 0).view(1,-1)
            vq_usage_entropy = torch_entropy(vq_usage).item()
            vq_usage = vq_usage / torch.sum(vq_usage)

            timer_epoch = time.time() - timer_epoch
            timer_curriculum = timer_curriculum / timer_epoch
            timer_forward = timer_forward / timer_epoch
            timer_backward = timer_backward / timer_epoch
            timer_remainder = 1. - timer_curriculum - timer_forward - timer_backward
            
            # report on loss
            if epoch % self.epoch2print == 0:
                speed = (time.time() - start_time) / (epoch + 1)
                print(f'\ntraining transformer ({self.transformer_type}) @ {self.run_name}, on {self.device}')
                print(f'... epoch {epoch}({epoch >= self.masking_patch_epoch}):\tcumloss = {np.round(cum_loss_total, 4)}\t{speed:0.4f} s/epoch')
                print(f'(1)...... loss rfb = {np.round(cum_loss_rfb, 4)} (mixing 1.)')
                print(f'(1)...... loss rfs = {np.round(cum_loss_rfs, 4)} (mixing {alpha_rfs})')
                print(f'(1)...... loss vq = {np.round(cum_loss_vq, 4)}')
                print(f'(1)...... loss sppech = {np.round(cum_loss_speech, 4)} (mixing {alpha_speech})')
                print(f'(2)...... avg. percent error RFB = {100 * avg_error_rfb:0.4f}\tRFS = {100 * avg_error_rfs:0.4f}')
                print(f'(3)...... current masking p = {masking_p:0.4f}')
                print(f'(4)...... vq usage stats: {(vq_usage.min().item(), vq_usage.mean().item(), vq_usage.max().item())}, H = {vq_usage_entropy:0.4f}')
                print('... timers:')
                print(f'...... curriculum: {timer_curriculum:0.4f}s')
                print(f'...... forward: {timer_forward:0.4f}s')
                print(f'...... backward: {timer_backward:0.4f}s')
                print(f'...... remainder: {timer_remainder:0.4f}s')
                print(f'dataset board proba: {self.dataset.board_types_proba}')

            # # test prediction with current model
            if epoch % self.epoch2test == 0:

                print('>>> computing accuracies on training and test sets...')

                masking_round = 2

                # figure out how many samples to take from training set
                nb_board_singleRoot = self.dataset.sets['singleRoot_test'].shape[0]
                nb_board_doubleRoot = self.dataset.sets['doubleRoot_test'].shape[0]
                nb_board_composite = self.dataset.sets['singleComposite_test'].shape[0]
                print(f'... nb_board_singleRoot: {nb_board_singleRoot}')
                print(f'... nb_board_doubleRoot: {nb_board_doubleRoot}')
                print(f'... nb_board_composite: {nb_board_composite}')

                # therefore
                nb_training_boards = masking_round * (nb_board_singleRoot + nb_board_doubleRoot + nb_board_composite)
                assert nb_training_boards > 0, 'test set is empty'
                board_types_proba = np.array([nb_board_singleRoot,
                                                           nb_board_doubleRoot,
                                                           nb_board_composite])
                board_types_proba = board_types_proba / np.sum(board_types_proba)

                board_types_proba_before = self.dataset.board_types_proba
                self.dataset.board_types_proba = board_types_proba

                # get accuracy on training set
                with torch.no_grad():
                    report_train = self.test_model_accuracy(nb_training_boards, 
                                                        masking_p = 0.2, 
                                                        patch_masking = False, 
                                                        patch_masking_p = 0.1, 
                                                        verbose = False,
                                                        display_erroneous_boards = False)
                
                # reset board_types_proba
                self.dataset.board_types_proba = board_types_proba_before
                
                # get accuracy on test set
                with torch.no_grad():
                    report_test = self.test_model_accuracy_set(target_set = 'test', 
                                                        masking_p = 0.2, 
                                                        patch_masking = False, 
                                                        patch_masking_p = 0.1,
                                                        masking_rounds=masking_round, 
                                                        verbose = False,
                                                        display_erroneous_boards = False)
                
                # report to wandb
                if self.bool_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "loss_total": cum_loss_total,
                        "loss_rfb": cum_loss_rfb,
                        "loss_rfs": cum_loss_rfs,
                        "loss_vq": cum_loss_vq,
                        "loss_speech": cum_loss_speech,
                        "avg_error_rfb": avg_error_rfb,
                        "avg_error_rfs": avg_error_rfs,
                        "vq_usage_mean": vq_usage.mean().item(), 
                        "vq_usage_entropy": vq_usage_entropy,
                        "board_types_proba": self.dataset.board_types_proba,
                        "nb_boards_test": nb_training_boards,

                        "train_acc_rfb_token": report_train['l0_acc_tokens'],
                        "train_acc_rfb_board": report_train['l0_acc_boards'],
                        "train_acc_rfb_breakdown_single": report_train['l0_acc_boards_breakdown'][0],
                        "train_acc_rfb_breakdown_double": report_train['l0_acc_boards_breakdown'][1],
                        "train_acc_rfb_breakdown_composite": report_train['l0_acc_boards_breakdown'][2],

                        "train_acc_rfs_token": report_train['l1_acc_tokens'],
                        "train_acc_rfs_board": report_train['l1_acc_boards'],
                        "train_acc_rfs_breakdown_single": report_train['l1_acc_boards_breakdown'][0],
                        "train_acc_rfs_breakdown_double": report_train['l1_acc_boards_breakdown'][1],
                        "train_acc_rfs_breakdown_composite": report_train['l1_acc_boards_breakdown'][2],

                        "test_acc_rfb_token": report_test['l0_acc_tokens'],
                        "test_acc_rfb_board": report_test['l0_acc_boards'],
                        "test_acc_rfb_breakdown_single": report_test['l0_acc_boards_breakdown'][0],
                        "test_acc_rfb_breakdown_double": report_test['l0_acc_boards_breakdown'][1],
                        "test_acc_rfb_breakdown_composite": report_test['l0_acc_boards_breakdown'][2],

                        "test_acc_rfs_token": report_test['l1_acc_tokens'],
                        "test_acc_rfs_board": report_test['l1_acc_boards'],
                        "test_acc_rfs_breakdown_single": report_test['l1_acc_boards_breakdown'][0],
                        "test_acc_rfs_breakdown_double": report_test['l1_acc_boards_breakdown'][1],
                        "test_acc_rfs_breakdown_composite": report_test['l1_acc_boards_breakdown'][2],
                        }, step=self.step)
                        
            # save current version of the network (save on the last epoch)
            if self.save_model and (epoch % self.epoch2save == 0 or (epoch + 1) == self.epochs):
                torch.save(self.transformer.state_dict(), f'./runs/{self.run_name}/checkpoints/model_{epoch}.pth')
                
        # print total duration of training
        total_time = time.time()-start_time
        total_time_min = total_time // 60
        total_time_sec = total_time - total_time_min * 60
        print(f'\n>> training completed in {total_time_min} min, {total_time_sec:.3f} s')
        
        return np.array(cum_losses)
    
    #..................................
    # forward passes, used to gather activations for analysis
    #..................................
    @torch.no_grad()
    def forward_activations_fromBoard(self, boards, masks, verbose = False):

        """
        Get activations while IN perform masked boards -> reconstructed boards.

        Args:
            boards (np.ndarray): (b, h, w) boards to be masked
            masks (np.ndarray): (b, h, w) masks indicating which tokens to mask
            verbose (bool): whether or not to print 
        Returns:
            activations_pca (dict): dictionary containing activations from the transformer
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

        # do forward pass from board inputs
        logits = self.transformer.predict_from_board(masked_boards) # (b,n*n,v)
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
        error = np.mean(np.abs(boards - boards_recons) > 0)
        print(f'reconstruction acc = {100 * (1- error):.2f}')

        return self.activations_pca

    @torch.no_grad()
    def forward_activations_fromSpeech(self, sentences, boards = None):
        
        """
        Get activations while IN perform speech -> reconstructed boards.

        Args:
            sentences (np.ndarray): (b, l) indices of words or (b, l, v) one-hot representation
            boards (np.ndarray): (b, h, w) reference boards to compute reconstruction error
        Returns:
            activations_pca (dict): dictionary containing activations from the transformer
        """

        sentences = torch.from_numpy(sentences).long().to(self.device) # one-hot representation

        if sentences.ndim == 3: # one-hot representation
            print(f'converting one-hot encodings to indices')
            sentences = torch.argmax(sentences, dim = -1) # convert to indices

        print(f'input sentences shape: {sentences.shape}')

        #..................................
        # register hooks & get activations
        #..................................
        # prepare hooks
        self._register_hooks_pca()
        self.activations_pca = {}

        # do forward pass from board inputs
        logits = self.transformer.predict_from_speech(sentences)
        boards_recons = torch.argmax(F.softmax(logits, dim = -1), dim = -1)

        self._unregister_hooks() # unregister hooks

        #..................................
        # convert all to numpy
        #..................................
        h = self.dataset.board_dim
        boards_recons = rearrange(boards_recons, 'b (h w) -> b h w', h = h).cpu().detach().numpy()

        for k, v in self.activations_pca.items():
            self.activations_pca[k] = v.cpu().detach().numpy()

        #..................................
        # compute cumulative error between original and reconstructed board
        #..................................
        if boards is not None:
            error = np.mean(np.abs(boards - boards_recons) > 0)
            print(f'reconstruction acc = {100 * (1- error):.2f}')

        return self.activations_pca

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
            report (dict): dictionary containing accuracy stats
        """

        # make boards
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
        
        # reshape and forward  
        boards = rearrange(boards, 'b h w -> b (h w)')
        masked_boards = rearrange(masked_boards, 'b h w -> b (h w)')
        ps_logits, ps2_logits, vq_loss, vq_encodings = self.transformer(masked_boards)

        report = {}
        # convert to numpy
        boards = boards.cpu().detach().numpy()
        masked_boards = masked_boards.cpu().detach().numpy()
        ps_logits = ps_logits.cpu().detach().numpy()
        ps2_logits = ps2_logits.cpu().detach().numpy()

        for idx_logit, logits in enumerate([ps_logits, ps2_logits]):

            # get reconstructed boards
            rec_boards = np.argmax(logits, axis = -1)

            # compute overall accuracy
            error_token = boards != rec_boards
            error_board = np.sum(error_token, axis = -1) > 0
            # find out which board types are erroneous
            error_board_types = board_types[error_board]

            acc_tokens = 1. - np.sum(error_token).item() / boards.size
            acc_boards = 1. - np.sum(error_board).item() / boards.shape[0]
            acc_boards_breakdown  = [
                1. - np.sum(error_board_types == 0) / np.sum(board_types == 0),
                1. - np.sum(error_board_types == 1) / np.sum(board_types == 1),
                1. - np.sum(error_board_types == 2) / np.sum(board_types == 2)
            ]

            if verbose:
                print(f'accuracy across tokens = {acc_tokens:.3f}')
                print(f'accuracy across boards = {acc_boards:.3f}')
                print(f'... single root: {np.sum(error_board_types == 0)}/{np.sum(board_types == 0)}, {acc_boards_breakdown[0]:.3f} acc')
                print(f'... double root: {np.sum(error_board_types == 1)}/{np.sum(board_types == 1)}, {acc_boards_breakdown[1]:.3f} acc')
                print(f'... composite {np.sum(error_board_types == 2)}/{np.sum(board_types == 2)}, {acc_boards_breakdown[2]:.3f} acc')
            
            # convert boards and masks to numpy
            display_boards = rearrange(boards, 'b (h w) -> b h w', h = self.dataset.board_dim)
            display_masks = np.zeros_like(masked_boards).astype(int)
            display_masks[masked_boards == self.masking_token] = 1
            display_masks = rearrange(display_masks, 'b (h w) -> b h w', h = self.dataset.board_dim)
            display_rec_boards = rearrange(rec_boards, 'b (h w) -> b h w', h = self.dataset.board_dim)

            # render boards with error
            if display_erroneous_boards:
                for idx_board in range(display_boards.shape[0]):
                    curr_error = np.sum(display_boards[idx_board] != display_rec_boards[idx_board])
                    if curr_error > 0:
                        board = display_boards[idx_board]
                        mask = display_masks[idx_board]
                        board_abs = -1 * np.ones_like(board).astype(np.int)
                        board_abs = np.where(board >= self.dataset.vocab_bck, 0, board_abs)

                        rec_board = display_rec_boards[idx_board]
                        rec_board_abs = -1 * np.ones_like(rec_board).astype(np.int)
                        rec_board_abs = np.where(rec_board >= self.dataset.vocab_bck, 0, rec_board_abs)

                        print('-'*50,f'\nboard {idx_board}: error = {curr_error}')
                        print('original:')
                        self.dataset.render_board(board, board_abs, mask=mask, figsize=(3,3), label_size=(8,0.15))
                        print('reconstructed:')
                        self.dataset.render_board(rec_board, rec_board_abs, mask=np.zeros_like(board), figsize=(3,3), label_size=(8,0.15))

            # write report
            report[f'l{idx_logit}_acc_tokens'] = acc_tokens
            report[f'l{idx_logit}_acc_boards'] = acc_boards
            report[f'l{idx_logit}_acc_boards_breakdown'] = acc_boards_breakdown

        return report

    # look at reconstruction error on a random set of masked boards from the specficied set (train, val, test)
    @torch.no_grad()
    def test_model_accuracy_set(self, target_set, masking_p, patch_masking = False, 
                            patch_masking_p = 0.1, masking_rounds = 1, verbose = False, display_erroneous_boards = False):
        
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

        Returns:
        - report (dict): dictionary containing accuracy stats
        """

        # make boards
        if masking_rounds > 1:
            boards = []
            board_types = []
            for _ in range(masking_rounds):
                boards_, board_types_ = self.dataset.get_all_boards(target_set)
                boards.append(boards_)
                board_types.append(board_types_)
            boards = np.concatenate(boards, axis = 0)
            board_types = np.concatenate(board_types, axis = 0)
        else:
            boards, board_types = self.dataset.get_all_boards(target_set)

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
        ps_logits, ps2_logits, vq_loss, vq_encodings = self.transformer(masked_boards)

        report = {}
        # convert to numpy
        boards = boards.cpu().detach().numpy()
        masked_boards = masked_boards.cpu().detach().numpy()
        ps_logits = ps_logits.cpu().detach().numpy()
        ps2_logits = ps2_logits.cpu().detach().numpy()

        for idx_logit, logits in enumerate([ps_logits, ps2_logits]):

            # get reconstructed boards
            rec_boards = np.argmax(logits, axis = -1)

            # compute overall accuracy
            error_token = boards != rec_boards
            error_board = np.sum(error_token, axis = -1) > 0
            # find out which board types are erroneous
            error_board_types = board_types[error_board]

            acc_tokens = 1. - np.sum(error_token).item() / boards.size
            acc_boards = 1. - np.sum(error_board).item() / boards.shape[0]
            acc_boards_breakdown  = [
                1. - np.sum(error_board_types == 0) / np.sum(board_types == 0),
                1. - np.sum(error_board_types == 1) / np.sum(board_types == 1),
                1. - np.sum(error_board_types == 2) / np.sum(board_types == 2)
            ]

            if verbose:
                print(f'accuracy across tokens = {acc_tokens:.3f}')
                print(f'accuracy across boards = {acc_boards:.3f}')
                print(f'... single root: {np.sum(error_board_types == 0)}/{np.sum(board_types == 0)}, {acc_boards_breakdown[0]:.3f} acc')
                print(f'... double root: {np.sum(error_board_types == 1)}/{np.sum(board_types == 1)}, {acc_boards_breakdown[1]:.3f} acc')
                print(f'... composite {np.sum(error_board_types == 2)}/{np.sum(board_types == 2)}, {acc_boards_breakdown[2]:.3f} acc')
            
            # convert boards and masks to numpy
            display_boards = rearrange(boards, 'b (h w) -> b h w', h = self.dataset.board_dim)
            display_masks = np.zeros_like(masked_boards).astype(int)
            display_masks[masked_boards == self.masking_token] = 1
            display_masks = rearrange(display_masks, 'b (h w) -> b h w', h = self.dataset.board_dim)
            display_rec_boards = rearrange(rec_boards, 'b (h w) -> b h w', h = self.dataset.board_dim)

            # render boards with error
            if display_erroneous_boards:
                for idx_board in range(display_boards.shape[0]):
                    curr_error = np.sum(display_boards[idx_board] != display_rec_boards[idx_board])
                    if curr_error > 0:
                        board = display_boards[idx_board]
                        mask = display_masks[idx_board]
                        board_abs = -1 * np.ones_like(board).astype(np.int)
                        board_abs = np.where(board >= self.dataset.vocab_bck, 0, board_abs)

                        rec_board = display_rec_boards[idx_board]
                        rec_board_abs = -1 * np.ones_like(rec_board).astype(np.int)
                        rec_board_abs = np.where(rec_board >= self.dataset.vocab_bck, 0, rec_board_abs)

                        print('-'*50,f'\nboard {idx_board}: error = {curr_error}')
                        print('original:')
                        self.dataset.render_board(board, board_abs, mask=mask, figsize=(3,3), label_size=(8,0.15))
                        print('reconstructed:')
                        self.dataset.render_board(rec_board, rec_board_abs, mask=np.zeros_like(board), figsize=(3,3), label_size=(8,0.15))

            # write report
            report[f'l{idx_logit}_acc_tokens'] = acc_tokens
            report[f'l{idx_logit}_acc_boards'] = acc_boards
            report[f'l{idx_logit}_acc_boards_breakdown'] = acc_boards_breakdown

        return report

    # look at reconstruction error on a specific set of masked boards
    @torch.no_grad()
    def test_model_accuracy_subset(self, boards, masks, bool_show_erroneous_boards = False, verbose = False):

        """
        test model accuracy on specified boards

        Args:
            boards (np.ndarray): boards to test on
            masks (np.ndarray): masks used to mask boards
            bool_show_erroneous_boards (bool): whether or not to display boards that are not reconstructed properly
        
        Returns:
            report (dict): dictionary containing accuracy stats
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
        ps_logits, ps2_logits, vq_loss, vq_encodings = self.transformer(masked_boards)

        report = {}
        # convert to numpy
        boards = boards.cpu().detach().numpy()
        masked_boards = masked_boards.cpu().detach().numpy()
        ps_logits = ps_logits.cpu().detach().numpy()
        ps2_logits = ps2_logits.cpu().detach().numpy()

        for idx_logit, logits in enumerate([ps_logits, ps2_logits]):

            # get reconstructed boards
            rec_boards = np.argmax(logits, axis = -1)

            # compute overall accuracy
            error_token = boards != rec_boards
            error_board = np.sum(error_token, axis = -1) > 0

            acc_tokens = 1. - np.sum(error_token).item() / boards.size
            acc_boards = 1. - np.sum(error_board).item() / boards.shape[0]

            if verbose:
                print(f'accuracy across tokens = {acc_tokens:.3f}')
                print(f'accuracy across boards = {acc_boards:.3f}')
            
            # convert boards and masks to numpy
            display_boards = rearrange(boards, 'b (h w) -> b h w', h = self.dataset.board_dim)
            display_masks = np.zeros_like(masked_boards).astype(int)
            display_masks[masked_boards == self.masking_token] = 1
            display_masks = rearrange(display_masks, 'b (h w) -> b h w', h = self.dataset.board_dim)
            display_rec_boards = rearrange(rec_boards, 'b (h w) -> b h w', h = self.dataset.board_dim)

            # render boards with error
            if bool_show_erroneous_boards:
                for idx_board in range(display_boards.shape[0]):
                    curr_error = np.sum(display_boards[idx_board] != display_rec_boards[idx_board])
                    if curr_error > 0:
                        board = display_boards[idx_board]
                        mask = display_masks[idx_board]
                        board_abs = -1 * np.ones_like(board).astype(np.int)
                        board_abs = np.where(board >= self.dataset.vocab_bck, 0, board_abs)

                        rec_board = display_rec_boards[idx_board]
                        rec_board_abs = -1 * np.ones_like(rec_board).astype(np.int)
                        rec_board_abs = np.where(rec_board >= self.dataset.vocab_bck, 0, rec_board_abs)

                        print('-'*50,f'\nboard {idx_board}: error = {curr_error}')
                        print('original:')
                        self.dataset.render_board(board, board_abs, mask=mask, figsize=(3,3), label_size=(8,0.15))
                        print('reconstructed:')
                        self.dataset.render_board(rec_board, rec_board_abs, mask=np.zeros_like(board), figsize=(3,3), label_size=(8,0.15))

            # write report
            report[f'l{idx_logit}_acc_tokens'] = acc_tokens
            report[f'l{idx_logit}_acc_boards'] = acc_boards

        return report

    #.................................................
    # speaker specific
    #.................................................

    @torch.no_grad()
    def speaker_predict(self, boards, masks):

        """
        Reconstruct masked boards from masked boards and speech.

        Args:
            boards (np.ndarray): (b, h, w) boards to be masked
            masks (np.ndarray): (b, h, w) masks indicating which tokens to mask
        Returns:
            rec_boards_fromPicture (np.ndarray): (b, h, w) reconstructed boards from picture
            rec_boards_fromSpeech (np.ndarray): (b, h, w) reconstructed boards from speech
            words (np.ndarray): (b, l, d) words used to reconstruct boards (one-hot encodings)
        """

        b, h, w = boards.shape

        #..................................
        # mask board as pth
        #..................................
        boards = torch.from_numpy(rearrange(boards,'b h w -> b (h w)')).long().to(self.device) # convert to tensor (k,n*n)
        masks = torch.from_numpy(rearrange(masks,'b h w -> b (h w)')).long().to(self.device) # convert to tensor (1,n*n)

        fully_masked_boards = self.masking_token * torch.ones(boards.shape, device = self.device)
        masked_boards = (masks * fully_masked_boards + (1-masks) * boards).long() # should be indices

        #..................................
        # pass through network as pth
        #..................................
        ps_logits, ps2_logits, _, vq_encodings = self.transformer(masked_boards)

        # convert to numpy
        boards = boards.cpu().detach().numpy()
        masked_boards = masked_boards.cpu().detach().numpy()
        ps_logits = ps_logits.cpu().detach().numpy()
        ps2_logits = ps2_logits.cpu().detach().numpy()
        words = vq_encodings.numpy()

        #..................................
        # reconstructions as np
        # compute error
        #..................................
        rec_boards_fromPicture = np.argmax(ps_logits, axis = -1)
        # compute overall accuracy
        error_token = boards != rec_boards_fromPicture
        error_board = np.sum(error_token, axis = -1) > 0
        acc_tokens = 1. - np.sum(error_token).item() / boards.size
        acc_boards = 1. - np.sum(error_board).item() / boards.shape[0]
        print(f'rec from board: acc tokens = {100 * acc_tokens:.2f} %, acc boards = {100 * acc_boards:.2f} %')

        rec_boards_fromSpeech = np.argmax(ps2_logits, axis = -1)
        # compute overall accuracy
        error_token = boards != rec_boards_fromSpeech
        error_board = np.sum(error_token, axis = -1) > 0
        acc_tokens = 1. - np.sum(error_token).item() / boards.size
        acc_boards = 1. - np.sum(error_board).item() / boards.shape[0]
        print(f'rec from speech: acc tokens = {100 * acc_tokens:.2f} %, acc boards = {100 * acc_boards:.2f} %')

        rec_boards_fromPicture = rearrange(rec_boards_fromPicture,'b (h w) -> b h w', h = h)
        rec_boards_fromSpeech = rearrange(rec_boards_fromSpeech,'b (h w) -> b h w', h = h)

        return rec_boards_fromPicture, rec_boards_fromSpeech, words
    
    @torch.no_grad()
    def speaker_predict_from_speech(self, sentences):
        """
        Predict boards from speech alone.

        Args:
            sentences (np.ndarray): (b, l) indices of words or (b, l, v) one-hot representation
        Returns:
            reconstructed_boards (np.ndarray): (b, h, w) reconstructed boards
        """
        sentences = torch.from_numpy(sentences).long().to(self.device) # one-hot representation

        if sentences.ndim == 3: # one-hot representation
            print(f'converting one-hot encodings to indices')
            sentences = torch.argmax(sentences, dim = -1) # convert to indices

        print(f'input sentences shape: {sentences.shape}')

        logits = self.transformer.predict_from_speech(sentences)
        reconstructed_boards = rearrange(torch.argmax(logits, dim = -1),'b (h w) -> b h w', h = self.dataset.board_dim).numpy()

        return reconstructed_boards

#/////////////////////////////////////////////////////////////////////////////////////

if __name__ == "__main__":
    pass