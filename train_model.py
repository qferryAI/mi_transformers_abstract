"""
Script to train a transformer model on the abstraction dataset.
The script is called with the following arguments:
    --device: device to be used for training (cpu or cuda)
    --epoch: number of epochs for training
    --type: type of transformer (vanilla or factor), note that vanilla is no longer supported
    --batch: number of times the code repeats1
    --save: 1 = save model, 0 = don't save model
"""

#----------------------------------------------------
# import libraries
#----------------------------------------------------
from datetime import date
import numpy as np
import os
import json
import argparse
from module_dataset import AbstractionDataset
from module_train_probe import TrainTransformer
#----------------------------------------------------

if __name__ == "__main__":
    
    #parse script arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type = str, default = 'cpu', help = "device to be used")
    parser.add_argument("--epoch", type = int, default = 1, help = "nb of epochs for training")
    parser.add_argument("--type", type = str, default = 'factor', help = "type of transformer")
    parser.add_argument("--batch", type = int, default = 1, help = "number of times the code repeats")
    parser.add_argument("--save", type = int, default = 0, help = "1 = save model, 0 = don't save model")
    args = parser.parse_args()
    
    transformer_type = args.type

    for _ in range(args.batch):

        #....................................................
        # create a folder for the run, 
        # requires folder `runs` in root directory
        #....................................................
        run_name = f'run_{date.today().strftime("%Y%m%d")}_{np.random.randint(100,1000)}'
        if transformer_type == 'factor':
            run_name += '_factor'

        #....................................................
        # set parameters for the dataset
        #....................................................
        params_dataset = {
            'dataset_len':1000, #nb of instances in the dataset
            'board_dim':8, #dim of board (board_dim, board_dim)
            'vocab_bck':10, #nb background tokens
            'vocab_token':10, #nb abstraction tokens (also called object tokens)
            'abs_n': 10, #nb of root abstractions (level #1 objects)
            'abs_dim':3, #footprint (abs_dim, abs_dim) of root abstractions (i.e., object)
            'abs_c':9, #cardinality: nb of tokens in the root abstraction (<= abs_dim**2)
            'abs_w_c':None, #nb of `wave` tokens (use None for deterministic abstractions) (<= abs_c) 
            'abs_w_m':1, #nb modes per `wave` token, default is 1
            'comp_n':5, #nb of compositions (composite abstractions, level #2 objects)
            'comp_dim':2, #dim of composition footprint
            'comp_margin':1, #margin between constituent root abstractions
            'comp_c':4, #nb of root in the comp_dim**2 pattern
            'board_types':['single', 'double', 'composition'], # types of boards to generate
            'board_types_proba': np.array([0.33, 0.33, 0.33]), # proba of generating each board type (single, double, composition)
            'run_name':run_name, # name of the run
            'card_val':0, # use fraction when split_method is 'fraction', otherwise nb of instances
            'card_test':4, # use fraction when split_method is 'fraction', otherwise nb of instances
            'split_method': 'balanced', # method is 'fraction', 'stringent', or 'balanced'
            'bool_split': True, # whether to split the dataset or not
        }
        #....................................................
        # create and train model
        #....................................................
        params_training = {
            'transformer_type': transformer_type, # should be 'factor'
            
            'vocab_size':params_dataset['vocab_bck'] + params_dataset['vocab_token'] + 1, # size of entire vocab (background + object + UNK)
            'block_size':params_dataset['board_dim']**2, # size of a context (board_dim**2)
            'n_layer': 3, # nb of transformer layers
            'n_head': 2, # nb of heads in multi-head attention
            'n_embd':64, # embedding dimensions
            'n_embd_pos':32, # positional embedding dimensions
            'embd_pdrop':0., # dropout rate for embeddings
            'resid_pdrop':0., # dropout rate for residual connections
            'attn_pdrop':0., # dropout rate for attention
            'n_unmasked':params_dataset['board_dim']**2, # depreciated
            
            'masking_token':params_dataset['vocab_bck'] + params_dataset['vocab_token'], # index of UNK. token
            'masking_p':0.25, # final masking probability
            'masking_patch_dim':3, # made obsolete
            'masking_patch_epoch':750, # epoch at which masking reach full proba & patch masking starts
            'masking_patch_p':0.5, # probability of patch masking for subsequent steps

            # new 06.23
            'board_curriculum': True, # whether to use curriculum learning
            'board_proba_epoch_start': 250, # epoch at which board curriculum starts
            'board_proba_epoch_end': 1000, # epoch at which board curriculum ends
            'board_proba_start': [0.8, 0.1, 0.1], #[0.8, 0.15, 0.05], # proba of generating each board type (single, double, composition) at start
            'board_proba_end': [0.33, 0.33, 0.33], # proba of generating each board type (single, double, composition) at end
            
            'learning_rate':1e-3, # learning rate
            'batch_size':64, # batch size
            'epochs': args.epoch, # total nb of epochs
            'device': args.device, # device to be used for training
            'plot_dpi': 150, # dpi for plots
                        
            'epoch2print': 25, # nb epoch between performance print
            'epoch2test': 250, # nb epoch between test (on training and test sets)

            'save_model': args.save == 1, # whether to save model or not
            'epoch2save': 250, # nb epoch between model save
            'run_name':run_name, # name of the run
        }


        # create folders tree: add additional folders
        for folderpath in [f'./runs/{run_name}/' + subdir for subdir in ['checkpoints', 'figures', 'datasets', 'analysis']]:
            try:
                os.makedirs(folderpath, exist_ok = True)
                print("Directory '%s' created successfully" % folderpath)
            except OSError as error:
                print("Directory '%s' can not be created" % folderpath)

        #....................................................
        # eport training parameters
        #....................................................       
        with open(f'./runs/{run_name}/params_training.json', 'w') as filename:
            json.dump(params_training, filename, indent = 4)
            
        #....................................................
        # make dataset and train
        #....................................................
        dataset = AbstractionDataset(**params_dataset)
        dataset.export(f'./runs/{run_name}/datasets/dataset')

        # print out some info about the dataset
        #dataset.whoamI() 

        train_transformer = TrainTransformer(dataset, **params_training)
        cum_losses = train_transformer.train()