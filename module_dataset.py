"""
Module containing the dataset class (AbstractionDataset).
"""

#----------------------------------------------------
# import libraries
#----------------------------------------------------
import math
import json
import numpy as np
from torch.utils.data import Dataset
from einops import rearrange
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
#----------------------------------------------------

#----------------------------------------------------
# class definition
#----------------------------------------------------
class AbstractionDataset(Dataset):

    """
    Inherit from the PyTorch Dataset class.

    * Dataset of synthetic (board_dim x board_dim) boards. Each board is a 2D array of tokens.
    
    * Each token is an integer between 0 and vocab_size-1, where vocab_size corresponds to the size of vocabulary:
    - The first vocab_bck tokens are background tokens. 
    - The next vocab_token tokens are object tokens.
    - The last token is the unknown token.

    * Object tokens can be composed to form root abstractions (specific objects = consistent set of obj. tokens aranged in a consitent way).
    * Root abstraction can be "fuzzy", meaning that certain token in the object can take more than one value (wave token). This allows to generate conditions where we can get multiple instances of a the same object.
    * Root abstractions can be composed to form composite abstractions (more complex objects = consistent set of root abstractions aranged in a consitent way).

    * Boards are created according to the following procedure:
    - choose a background token and populate the entire board with it.
    - choose a board type (single, double, composition) and populate the board with the corresponding objects.
        - single: Add single root abstraction to the board (abstraction ID and position are random).
        - double: Add two root abstractions (abstraction IDs and positions are random).
        - composition: Add a single composite abstraction to the board (abstraction ID and position are random).
    - add masking before returning the board.

    See __init__ for full details on the arguments.
    """
    
    def __init__(self, **kwargs):

        """
        Initializes the AbstractionDataset class. Using parameters detailed below, it generates a set of root and composite abstractions, which can then be used to produce boards. If 'self.bool_split' is True, all possible boards are created and split into train, val and test sets. 

        Args:
            'dataset_len':1000, #nb of instances in the dataset (only relevant for torch DataLoader)
            'board_dim':8, #dimension of board (height and width). Boards.shape will be (board_dim, board_dim)
            'vocab_bck':10, #nb background tokens
            'vocab_token':10, #nb abstraction tokens (also called object tokens)
            'abs_n': 10, #nb of root abstractions (level #1 objects)
            'abs_dim':3, #footprint (abs_dim, abs_dim) of root abstractions
            'abs_c':9, #cardinality: nb of object tokens in the footprint that will make up the abstraction (abs_c <= abs_dim**2)
            'abs_w_c':None, #nb of fuzzy tokens (use None for deterministic abstractions) amongst the abs_c tokens of the abstraction (abs_w_c <= abs_c) 
            'abs_w_m':1, #nb modes per fuzzy token, default is 1
            'comp_n':5, #nb of composite abstractions (level #2 objects)
            'comp_dim':2, #footprint (comp_dim, comp_dim) of composite abstractions
            'comp_margin':1, #margin between constituent root abstractions
            'comp_c':4, #cardinality: nb of constituent root abstractions in the composite footprint (comp_c <= comp_dim**2)
            'board_types':['single', 'double', 'composition'], #types of boards to generate
            'board_types_proba': np.array([0.33, 0.33, 0.33]), #proba of generating each board type (single, double, composition)
            'run_name':run_name, #name of the run (used for saving)
            'bool_split': True,# whether to split the dataset or not
            'split_method': 'balanced', # method is 'fraction', 'stringent', or 'balanced'
            'card_val':0, # use fraction when split_method is 'fraction', otherwise nb of instances
            'card_test':4, # use fraction when split_method is 'fraction', otherwise nb of instances

        Returns:
        An instance of the AbstractionDataset class.
        """

        print('Creating an instance of the AbstractionDataset class')

        super(AbstractionDataset, self).__init__()
        
        for k, v in kwargs.items():
            setattr(self,k,v) # set attribute to self
            
        self._make_root_abstractions() # make root abstractions
        self._make_compositions() # make composite abstractions

        # check that card_val and card_test exist as arguments
        # for legacy reasons, these arguments are not always present
        if not hasattr(self, 'card_val'):
            self.card_val = 0
        if not hasattr(self, 'card_test'):
            self.card_test = 0
        if not hasattr(self, 'split_method'):
            self.split_method = 'balanced'
        if not hasattr(self, 'bool_split'):
            self.bool_split = False
        
        # split dataset into train, val and test sets
        if self.split_method == 'fraction' and self.bool_split:
            print('splitting dataset into train, val and test sets, method: fraction')
            self._split_sets_fraction()
        elif self.split_method == 'stringent' and self.bool_split:
            print('splitting dataset into train, val and test sets, method: stringent')
            self._split_sets_stringent()
        elif self.split_method == 'balanced' and self.bool_split:
            print('splitting dataset into train, val and test sets, method: balanced')
            self._split_sets_balanced()
        else:
            print('not splitting dataset into train, val and test sets')
            self.sets = None

        self.set_type = 'train' # 'train', 'val', 'test' set current set. Allows to draw boards from a specific set.

    #//////////////////////////////////////////////////////
    # EXPORT / IMPORT
    # use to save and load dataset parameters to and from file
    # ////////////////////////////////////////////////////// 
        
    def export(self, filename):

        """
        Exports dataset parameters and splits to the file specified by 'filename'.

        Args:
            'filename' (str): name of the file to export to (without extension)
        
        Returns:
            None
        """
        
        dataset_vars = {} # dict to store dataset parameters

        # save dataset parameters to dict        
        for k, v in vars(self).items():
            if type(v).__module__ == np.__name__: # if current parameter is a np.array
                # create a file and save np.array
                array_filename = f'{filename}_{k}.npy' 
                np.save(array_filename, v)
                # replace with filename in dict
                dataset_vars[k] = array_filename
            else:
                if k in ['sets'] and v is not None:
                    # convert sets to dict of list
                    sets_lists = {}
                    for k_set, v_set in v.items():
                        sets_lists[k_set] = v_set.tolist()
                    dataset_vars[k] = sets_lists
                else:
                    dataset_vars[k] = v

        # save dataset_vars to disc as json
        with open(f'{filename}.json', 'w') as json_file:
            json.dump(dataset_vars, json_file, indent = 4)
    
    def load(self, filename):

        """
        load dataset parameters from the file specified by 'filename'.

        Args:
            'filename' (str): name of the file to load from (without extension)
        
        Returns:
            None
        """
        
        # read parameters from json file
        with open(f'{filename}.json') as json_file:
            dataset_vars = json.load(json_file)

        # test if dataset_vars contains card_val, card_test, and set_type
        # for legacy reasons, these arguments are not always present
        # if not, set them to 0
        if not 'card_val' in dataset_vars:
            dataset_vars['card_val'] = 0
        if not 'card_test' in dataset_vars:
            dataset_vars['card_test'] = 0 
        if not 'set_type' in dataset_vars:
            dataset_vars['set_type'] = 'train' 
        if not 'split_method' in dataset_vars:
            dataset_vars['split_method'] = 'balanced'
        if not 'bool_split' in dataset_vars:
            dataset_vars['bool_split'] = False
        
        # set dataset parameters
        for k, v in vars(self).items():
            if k in ['board_types_proba', 'roots', 'compositions']:
                array_filename = dataset_vars[k]
                setattr(self,k,np.load(array_filename))
            elif k in ['sets'] and dataset_vars[k] is not None:
                if 'sets' in dataset_vars:
                    # convert sets to dict of np.array
                    sets_arrays = {}
                    for k_set, v_set in dataset_vars[k].items():
                        sets_arrays[k_set] = np.array(v_set)
                    setattr(self,k,sets_arrays)
            else:
                setattr(self,k,dataset_vars[k])        
    
    #//////////////////////////////////////////////////////
    # MAKING ABSTRACTIONS
    # //////////////////////////////////////////////////////
    
    def _make_root_abstractions(self):

        """
        Creates root abstractions and updates self.roots.

        Args:
            None
        Returns:
            None
        """
        
        # check that parameters are valid
        assert self.abs_n >= 1, "[making root abstractions] error: abs_n < 1"
        assert self.abs_c <= self.abs_dim**2, "[making root abstractions] error: abs_c > abs_dim**2"
        
        # if not creating fuzzy objects, set abs_w_m to 1
        if self.abs_w_c is None:
            self.abs_w_m = 1
        
        # store root abstractions
        self.roots = -1 * np.ones((self.abs_n, self.abs_dim, self.abs_dim, self.abs_w_m))
        
        for idx_abs in range(self.abs_n):

            bool_unique = False # ensure uniqueness of root abstractions

            while(not bool_unique):
            
                # choose abs_c pixel for the abstraction
                tokens = np.random.choice(self.abs_dim**2,(self.abs_c,), replace = False)
                
                # choose abs_w_c fuzzy token
                if self.abs_w_c is None:
                    wave_tokens = []
                else:
                    assert self.abs_w_c <= self.abs_c, "[making root abstractions] error: abs_w_c > abs_c"
                    wave_tokens =  np.random.choice(tokens,(self.abs_w_c,), replace = False)
                
                # create abstraction, default value is -1
                curr_abs = -1 * np.ones((self.abs_dim**2, self.abs_w_m))
                
                for idx_token in list(tokens):
                    if idx_token in list(wave_tokens):
                        # choose abs_w_m tokens from vocab token
                        curr_abs[idx_token,:] = np.random.choice(np.arange(self.vocab_token)+self.vocab_bck, (self.abs_w_m,), replace = False)
                    else:
                        # choose a single token from vocab token
                        curr_abs[idx_token,0] = np.random.choice(np.arange(self.vocab_token)+self.vocab_bck, (1,)).item()
                
                # test uniqueness
                bool_unique = True
                for i in range(idx_abs-1):
                    if np.array_equal(curr_abs, self.roots[i]):
                        bool_unique = False
            
            # store root abstraction
            self.roots[idx_abs] = np.copy(rearrange(curr_abs, '(n1 n2) m -> n1 n2 m', n1 = self.abs_dim))    
    
        self.roots = self.roots.astype(int) # convert to int
        
    def _make_compositions(self):

        """
        Creates composite abstractions and updates self.compositions.

        Args:
            None
        Returns:
            None
        """
        
        assert self.comp_c <= self.comp_dim**2, "[making compositions] error: comp_c > comp_dim**2]"
        assert self.comp_dim * self.abs_dim + (self.comp_dim - 1) * self.comp_margin <= self.board_dim, "[making compositions] error: comp_dim * abs_dim + (comp_dim - 1) * comp_margin > board_dim"
        
        # store compositions
        self.compositions = np.zeros((self.comp_n, self.comp_dim, self.comp_dim))
        
        for idx_composition in range(self.comp_n):

            bool_unique = False

            while(not bool_unique):
            
                abstractions = np.arange(self.comp_dim**2)
                # choose position for comp_c abstractions
                abstractions = np.random.choice(abstractions,(self.comp_c,), replace = False)
                
                curr_composition = -1 * np.ones((self.comp_dim**2,))
                curr_composition[abstractions] = np.random.choice(self.abs_n, (self.comp_c,), replace = True)

            # test uniqueness
                bool_unique = True
                for i in range(idx_composition-1):
                    if np.array_equal(curr_composition, self.compositions[i]):
                        bool_unique = False

            self.compositions[idx_composition] = rearrange(curr_composition, '(n1 n2) -> n1 n2', n1 = self.comp_dim)
        
        self.compositions = self.compositions.astype(int)
    
    #//////////////////////////////////////////////////////
    # SPLIT DATASET
    # //////////////////////////////////////////////////////
    
    def _split_sets_balanced(self, verbose = True):

        """
        split the dataset into train, val and test sets in a balanced way, i.e., the different sets will have the same number of boards matching a specifict type (single, double, composite) and featuring specific abstraction(s):

        * For single root abstractions, the function generate all possible boards with a specific abstraction and allocate `card_val` boards to the validation set, `card_test` boards to the test set and the rest to the training set.

        * For double root abstractions, the function generate all possible boards with a specific pair of abstractions and allocate `card_val` boards to the validation set, `card_test` boards to the test set and the rest to the training set.

        * For single composite abstractions, the function generate all possible boards with a specific abstraction and allocate `card_val` boards to the validation set, `card_test` boards to the test set and the rest to the training set.

        Updates self.sets.

        Args:
            None
        Returns:
            None
        """

        self.sets = {}
        set_counter_dict = {'train': 0, 'val': 0, 'test': 0}

        #........................................
        # make all single root abstraction boards
        #........................................

        list_singleRoot_val = []
        list_singleRoot_test = []
        list_singleRoot_train = []

        nb_positions = self.board_dim - self.abs_dim + 1

        for idx_root in range(self.abs_n):
            curr_list = []
            for idx_x in range(nb_positions):
                for idx_y in range(nb_positions):
                    for idx_bck in range(self.vocab_bck):
                        curr_list.append(np.array([idx_root, idx_x, idx_y, idx_bck]).reshape(1, -1))

            curr_ndarray = np.concatenate(curr_list, axis=0)
            np.random.shuffle(curr_ndarray)

            list_singleRoot_val.append(curr_ndarray[:self.card_val])
            list_singleRoot_test.append(curr_ndarray[self.card_val:self.card_val+self.card_test])
            list_singleRoot_train.append(curr_ndarray[self.card_val+self.card_test:])

        self.sets['singleRoot_val'] = np.concatenate(list_singleRoot_val, axis=0)
        self.sets['singleRoot_test'] = np.concatenate(list_singleRoot_test, axis=0)
        self.sets['singleRoot_train'] = np.concatenate(list_singleRoot_train, axis=0)

        set_counter_dict['val'] += self.sets['singleRoot_val'].shape[0]
        set_counter_dict['test'] += self.sets['singleRoot_test'].shape[0]
        set_counter_dict['train'] += self.sets['singleRoot_train'].shape[0]

        #........................................
        # double roots
        #........................................

        list_doubleRoot_val = []
        list_doubleRoot_test = []
        list_doubleRoot_train = []

        for idx_root_1 in range(self.abs_n):
            for idx_root_2 in range(self.abs_n):
                curr_list = []
                for idx_x_1 in range(nb_positions):
                    for idx_y_1 in range(nb_positions):
                        for idx_x_2 in range(nb_positions):
                            for idx_y_2 in range(nb_positions):
                                for idx_bck in range(self.vocab_bck):
                                    if idx_x_1 != idx_x_2 and idx_y_1 != idx_y_2:
                                        curr_list.append(np.array([idx_root_1, idx_x_1, idx_y_1, idx_root_2, idx_x_2, idx_y_2, idx_bck]).reshape(1, -1))

                curr_ndarray = np.concatenate(curr_list, axis=0)
                np.random.shuffle(curr_ndarray)

                list_doubleRoot_val.append(curr_ndarray[:self.card_val])
                list_doubleRoot_test.append(curr_ndarray[self.card_val:self.card_val+self.card_test])
                list_doubleRoot_train.append(curr_ndarray[self.card_val+self.card_test:])

        self.sets['doubleRoot_val'] = np.concatenate(list_doubleRoot_val, axis=0)
        self.sets['doubleRoot_test'] = np.concatenate(list_doubleRoot_test, axis=0)
        self.sets['doubleRoot_train'] = np.concatenate(list_doubleRoot_train, axis=0)

        set_counter_dict['val'] += self.sets['doubleRoot_val'].shape[0]
        set_counter_dict['test'] += self.sets['doubleRoot_test'].shape[0]
        set_counter_dict['train'] += self.sets['doubleRoot_train'].shape[0]

        # ........................................
        # single composite
        # ........................................

        list_singleComposite_val = []
        list_singleComposite_test = []
        list_singleComposite_train = []

        nb_positions = self.board_dim - (self.comp_dim * self.abs_dim + (self.comp_dim - 1) * self.comp_margin) + 1

        for idx_comp in range(self.comp_n):
            curr_list = []
            for idx_x in range(nb_positions):
                for idx_y in range(nb_positions):
                    for idx_bck in range(self.vocab_bck):
                        curr_list.append(np.array([idx_comp, idx_x, idx_y, idx_bck]).reshape(1, -1))

            curr_ndarray = np.concatenate(curr_list, axis=0)
            np.random.shuffle(curr_ndarray)

            list_singleComposite_val.append(curr_ndarray[:self.card_val])
            list_singleComposite_test.append(curr_ndarray[self.card_val:self.card_val+self.card_test])
            list_singleComposite_train.append(curr_ndarray[self.card_val+self.card_test:])

        self.sets['singleComposite_val'] = np.concatenate(list_singleComposite_val, axis=0)
        self.sets['singleComposite_test'] = np.concatenate(list_singleComposite_test, axis=0)
        self.sets['singleComposite_train'] = np.concatenate(list_singleComposite_train, axis=0)

        set_counter_dict['val'] += self.sets['singleComposite_val'].shape[0]
        set_counter_dict['test'] += self.sets['singleComposite_test'].shape[0]
        set_counter_dict['train'] += self.sets['singleComposite_train'].shape[0]

        # print content of sets
        if verbose:
            for key in self.sets.keys():
                print(key, self.sets[key].shape)
            
            print(set_counter_dict)

    def _split_sets_fraction(self, verbose = True):

        """

        Create a split of the dataset into train, val and test sets based on parameters card_val and card_test.
        * Functions generate all possible boards with a single root abstraction and allocate `card_val` boards to the validation set, `card_test` boards to the test set and the rest to the training set. It then does the same for double root abstractions and single composite abstractions. 

        * split the dataset into train, val and test sets
        - self.card_val: fraction of all boards for validation set
        - self.card_test: fraction of all boards for test set
        - Training set gets 1. - self.card_val - self.card_test of all boards in a given category (single, double, composite).

        Updates self.sets.

        Args:
            None
        Returns:
            None
        """

        self.sets = {}
        set_counter_dict = {'train': 0, 'val': 0, 'test': 0}

        #........................................
        # single root
        #........................................

        list_singleRoot_all = []
        nb_positions = self.board_dim - self.abs_dim + 1

        for idx_root in range(self.abs_n):
            for idx_x in range(nb_positions):
                for idx_y in range(nb_positions):
                    for idx_bck in range(self.vocab_bck):
                        list_singleRoot_all.append(np.array([idx_root, idx_x, idx_y, idx_bck]).reshape(1, -1))

        # concatenate all
        singleRoot_all = np.concatenate(list_singleRoot_all, axis=0)

        # sample randomly to create val, test and train sets
        nb_boards = singleRoot_all.shape[0]
        nb_val = math.floor(nb_boards * self.card_val)
        nb_test = math.floor(nb_boards * self.card_test)
        # shuffle
        np.random.shuffle(singleRoot_all)
        # split

        self.sets['singleRoot_val'] = singleRoot_all[:nb_val]
        self.sets['singleRoot_test'] = singleRoot_all[nb_val:nb_val+nb_test]
        self.sets['singleRoot_train'] = singleRoot_all[nb_val+nb_test:]

        set_counter_dict['val'] += self.sets['singleRoot_val'].shape[0]
        set_counter_dict['test'] += self.sets['singleRoot_test'].shape[0]
        set_counter_dict['train'] += self.sets['singleRoot_train'].shape[0]

        #........................................
        # double roots
        #........................................

        list_doubleRoot_all = []

        for idx_root_1 in range(self.abs_n):
            for idx_root_2 in range(self.abs_n):
                for idx_x_1 in range(nb_positions):
                    for idx_y_1 in range(nb_positions):
                        for idx_x_2 in range(nb_positions):
                            for idx_y_2 in range(nb_positions):
                                for idx_bck in range(self.vocab_bck):
                                    if idx_x_1 != idx_x_2 and idx_y_1 != idx_y_2:
                                        list_doubleRoot_all.append(np.array([idx_root_1, idx_x_1, idx_y_1, idx_root_2, idx_x_2, idx_y_2, idx_bck]).reshape(1, -1))

        # concatenate all
        doubleRoot_all = np.concatenate(list_doubleRoot_all, axis=0)

        # sample randomly to create val, test and train sets
        nb_boards = doubleRoot_all.shape[0]
        nb_val = math.floor(nb_boards * self.card_val)
        nb_test = math.floor(nb_boards * self.card_test)
        # shuffle
        np.random.shuffle(doubleRoot_all)
        # split
        self.sets['doubleRoot_val'] = doubleRoot_all[:nb_val]
        self.sets['doubleRoot_test'] = doubleRoot_all[nb_val:nb_val+nb_test]
        self.sets['doubleRoot_train'] = doubleRoot_all[nb_val+nb_test:]

        set_counter_dict['val'] += self.sets['doubleRoot_val'].shape[0]
        set_counter_dict['test'] += self.sets['doubleRoot_test'].shape[0]
        set_counter_dict['train'] += self.sets['doubleRoot_train'].shape[0]

        # ........................................
        # single composite
        # ........................................

        list_singleComposite_all = []

        nb_positions = self.board_dim - (self.comp_dim * self.abs_dim + (self.comp_dim - 1) * self.comp_margin) + 1

        for idx_comp in range(self.comp_n):
            for idx_x in range(nb_positions):
                for idx_y in range(nb_positions):
                    for idx_bck in range(self.vocab_bck):
                        list_singleComposite_all.append(np.array([idx_comp, idx_x, idx_y, idx_bck]).reshape(1, -1))

        # concatenate all
        singleComposite_all = np.concatenate(list_singleComposite_all, axis=0)

        # sample randomly to create val, test and train sets
        nb_boards = singleComposite_all.shape[0]
        nb_val = math.floor(nb_boards * self.card_val)
        nb_test = math.floor(nb_boards * self.card_test)
        # shuffle
        np.random.shuffle(singleComposite_all)
        # split
        self.sets['singleComposite_val'] = singleComposite_all[:nb_val]
        self.sets['singleComposite_test'] = singleComposite_all[nb_val:nb_val+nb_test]
        self.sets['singleComposite_train'] = singleComposite_all[nb_val+nb_test:]

        set_counter_dict['val'] += self.sets['singleComposite_val'].shape[0]
        set_counter_dict['test'] += self.sets['singleComposite_test'].shape[0]
        set_counter_dict['train'] += self.sets['singleComposite_train'].shape[0]

        # print content of sets
        if verbose:
            for key in self.sets.keys():
                print(key, self.sets[key].shape)
            
            print(set_counter_dict)

    def _split_sets_stringent(self, verbose = True):

        """
        Implements a more stringent split of the dataset into train, val and test sets.
        The function ensures that for each abstraction, there exist a position on the board such that the abstraction is never presented at that position in the training data. The held out boards are devided equally between the validation and test sets.

        Updates self.sets.

        Args:
            None
        Returns:
            None
        """

        self.sets = {}
        set_counter_dict = {'train': 0, 'val': 0, 'test': 0}

        #........................................
        # single root
        #........................................
        list_singleRoot_train = []
        list_singleRoot_heldOut = []

        nb_positions = self.board_dim - self.abs_dim + 1

        # get on held out position for each root abstraction
        heldOut_position = np.random.choice(nb_positions, (self.abs_n,2), replace = True).astype(int)

        for idx_root in range(self.abs_n):
            curr_list = []
            curr_list_heldOut = []

            for idx_x in range(nb_positions):
                for idx_y in range(nb_positions):
                    for idx_bck in range(self.vocab_bck):
                        if idx_x == heldOut_position[idx_root, 0] and idx_y == heldOut_position[idx_root, 1]:
                            curr_list_heldOut.append(np.array([idx_root, idx_x, idx_y, idx_bck]).reshape(1, -1))
                        else:
                            curr_list.append(np.array([idx_root, idx_x, idx_y, idx_bck]).reshape(1, -1))

            curr_ndarray = np.concatenate(curr_list, axis=0)
            curr_ndarray_heldOut = np.concatenate(curr_list_heldOut, axis=0)

            # assert that curr_ndarray_heldOut is not empty
            assert curr_ndarray_heldOut.shape[0] > 0, "[making single root] error: curr_ndarray_heldOut is empty"

            list_singleRoot_train.append(curr_ndarray)
            list_singleRoot_heldOut.append(curr_ndarray_heldOut)

        
        list_singleRoot_train = np.concatenate(list_singleRoot_train, axis=0)
        list_singleRoot_heldOut = np.concatenate(list_singleRoot_heldOut, axis=0)

        #........................................
        # double roots
        #........................................
        list_doubleRoot_train = []
        list_doubleRoot_heldOut = []

        for idx_root_1 in range(self.abs_n):
            for idx_root_2 in range(self.abs_n):
                curr_list = []
                curr_list_heldOut = []
                for idx_x_1 in range(nb_positions):
                    for idx_y_1 in range(nb_positions):
                        for idx_x_2 in range(nb_positions):
                            for idx_y_2 in range(nb_positions):
                                if idx_x_1 != idx_x_2 and idx_y_1 != idx_y_2:
                                    for idx_bck in range(self.vocab_bck):
                                        if (idx_x_1 == heldOut_position[idx_root_1, 0] and idx_y_1 == heldOut_position[idx_root_1, 1]) or (idx_x_2 == heldOut_position[idx_root_2, 0] and idx_y_2 == heldOut_position[idx_root_2, 1]):
                                            curr_list_heldOut.append(np.array([idx_root_1, idx_x_1, idx_y_1, idx_root_2, idx_x_2, idx_y_2, idx_bck]).reshape(1, -1))
                                        else:
                                            curr_list.append(np.array([idx_root_1, idx_x_1, idx_y_1, idx_root_2, idx_x_2, idx_y_2, idx_bck]).reshape(1, -1))

                curr_ndarray = np.concatenate(curr_list, axis=0)
                list_doubleRoot_train.append(curr_ndarray)

                if len(curr_list_heldOut) > 0:
                    curr_ndarray_heldOut = np.concatenate(curr_list_heldOut, axis=0)
                    list_doubleRoot_heldOut.append(curr_ndarray_heldOut)
        
        list_doubleRoot_train = np.concatenate(list_doubleRoot_train, axis=0)
        list_doubleRoot_heldOut = np.concatenate(list_doubleRoot_heldOut, axis=0)

        # ........................................
        # single composite
        # ........................................

        list_singleComposite_train = []
        list_singleComposite_heldOut = []

        nb_positions = self.board_dim - (self.comp_dim * self.abs_dim + (self.comp_dim - 1) * self.comp_margin) + 1

        for idx_comp in range(self.comp_n):
            curr_list = []
            curr_list_heldOut = []

            for idx_x in range(nb_positions):
                for idx_y in range(nb_positions):

                    # check to see if held out position is in the current composite
                    curr_comp = self.compositions[idx_comp]
                    bool_heldOut = False

                    for i in range(self.comp_dim):
                        for j in range(self.comp_dim):
                            curr_root = curr_comp[i,j]
                            curr_root_position = np.array([idx_x + i * (self.abs_dim + self.comp_margin), idx_y + j * (self.abs_dim + self.comp_margin)])
                            if np.array_equal(curr_root_position, heldOut_position[curr_root]):
                                bool_heldOut = True
                                print(curr_root, curr_root_position, heldOut_position[curr_root], bool_heldOut)

                    for idx_bck in range(self.vocab_bck):
                        if bool_heldOut:
                            curr_list_heldOut.append(np.array([idx_comp, idx_x, idx_y, idx_bck]).reshape(1, -1))
                        else:
                            curr_list.append(np.array([idx_comp, idx_x, idx_y, idx_bck]).reshape(1, -1))

            curr_ndarray = np.concatenate(curr_list, axis=0)
            list_singleComposite_train.append(curr_ndarray)

            if len(curr_list_heldOut) > 0:
                curr_ndarray_heldOut = np.concatenate(curr_list_heldOut, axis=0)
                list_singleComposite_heldOut.append(curr_ndarray_heldOut)
        
        list_singleComposite_train = np.concatenate(list_singleComposite_train, axis=0)
        if len(list_singleComposite_heldOut) > 0:
            list_singleComposite_heldOut = np.concatenate(list_singleComposite_heldOut, axis=0)
        else:
            list_singleComposite_heldOut = np.zeros((0,4))

        # ........................................
        # split and create sets
        # ........................................

        # shuffle
        np.random.shuffle(list_singleRoot_train)
        np.random.shuffle(list_singleRoot_heldOut)

        n = list_singleRoot_heldOut.shape[0]
        if self.card_val > 0:
            n_val = int(n/2)
        else:
            n_val = 0

        self.sets['singleRoot_val'] = list_singleRoot_heldOut[:n_val]
        self.sets['singleRoot_test'] = list_singleRoot_heldOut[n_val:]
        self.sets['singleRoot_train'] = list_singleRoot_train

        set_counter_dict['val'] += self.sets['singleRoot_val'].shape[0]
        set_counter_dict['test'] += self.sets['singleRoot_test'].shape[0]
        set_counter_dict['train'] += self.sets['singleRoot_train'].shape[0]
        #....................................................................

        # shuffle
        np.random.shuffle(list_doubleRoot_train)
        np.random.shuffle(list_doubleRoot_heldOut)

        n = list_doubleRoot_heldOut.shape[0]
        if self.card_val > 0:
            n_val = int(n/2)
        else:
            n_val = 0

        self.sets['doubleRoot_val'] = list_doubleRoot_heldOut[:n_val]
        self.sets['doubleRoot_test'] = list_doubleRoot_heldOut[n_val:]
        self.sets['doubleRoot_train'] = list_doubleRoot_train

        set_counter_dict['val'] += self.sets['doubleRoot_val'].shape[0]
        set_counter_dict['test'] += self.sets['doubleRoot_test'].shape[0]
        set_counter_dict['train'] += self.sets['doubleRoot_train'].shape[0]
        #....................................................................

        # shuffle
        np.random.shuffle(list_singleComposite_train)
        np.random.shuffle(list_singleComposite_heldOut)

        n = list_singleComposite_heldOut.shape[0]
        if self.card_val > 0:
            n_val = int(n/2)
        else:
            n_val = 0

        self.sets['singleComposite_val'] = list_singleComposite_heldOut[:n_val]
        self.sets['singleComposite_test'] = list_singleComposite_heldOut[n_val:]
        self.sets['singleComposite_train'] = list_singleComposite_train

        set_counter_dict['val'] += self.sets['singleComposite_val'].shape[0]
        set_counter_dict['test'] += self.sets['singleComposite_test'].shape[0]
        set_counter_dict['train'] += self.sets['singleComposite_train'].shape[0]
        #....................................................................

        # print content of sets
        if verbose:
            for key in self.sets.keys():
                print(key, self.sets[key].shape)
            
            print(set_counter_dict)
    
    #//////////////////////////////////////////////////////
    # DRAWING BOARDS
    # /////////////////////////////////////////////////////
    
    def _draw_root_abstraction(self, idx_abs):

        """
        Draws a root abstraction from idx_abs, the index of the object in the root abstractions list.

        Args:
            'idx_abs' (int): index of the root abstraction to draw.
        Returns:
            'curr_abs_collapse' (np.array): matrix representing the abstraction.
        """
        
        curr_abs = np.copy(self.roots[idx_abs]) # include wave tokens
        curr_abs_collapse = np.zeros((self.abs_dim, self.abs_dim)) # with collapse wave tokens

        for i in range(self.abs_dim):
            for j in range(self.abs_dim):
                if np.sum(curr_abs[i,j,:] == -1) == self.abs_w_m: # no wave tokens at this position
                    curr_abs_collapse[i,j] = -1
                elif np.sum(curr_abs[i,j,:] != -1) == 1: # only one mode
                    curr_abs_collapse[i,j] = curr_abs[i,j,0]
                else: # multiple modes at this position
                    mode = np.random.choice(self.abs_w_m, (1,)).astype(int).item() # choose a mode randomly
                    curr_abs_collapse[i,j] = curr_abs[i,j,mode]
        
        return curr_abs_collapse # returns collapsed abstraction
    
    def _draw_composite(self, C, margin = 0):

        """
        draw a composite abstraction from C, the composition matrix and margin, the margin between constituent root abstractions.

        Args:
            'C' (np.array): composition matrix
            'margin' (int): margin between constituent root abstractions.
        Returns:
            'curr_composite' (np.array): matrix representing the composite abstraction.
            'curr_abs_mask' (np.array): matrix indicating which root abstraction is at each position.
        """
        
        C = C.astype(int)
        
        # C has index of root abstractions or -1
        n = C.shape[0]
        N = n*self.abs_dim + (n-1) * margin
        
        curr_composite = -1 * np.ones((N, N)) # give tokens at position
        curr_abs_mask = -1 * np.ones((N, N)) # indicates which root abstraction is at each abstract position
        
        for i in range(n):
            for j in range(n):
                if C[i,j] != -1: # if there is a root abstraction at this position
                    
                    # location of the root abstraction in the composite footprint
                    i_s = i * (self.abs_dim + margin)
                    i_e = i_s + self.abs_dim
                    j_s = j * (self.abs_dim + margin)
                    j_e = j_s + self.abs_dim
                    
                    curr_composite[i_s:i_e, j_s:j_e] = self._draw_root_abstraction(C[i,j])
                    curr_abs_mask[i_s:i_e, j_s:j_e] = C[i,j]
        
        return curr_composite.astype(int), curr_abs_mask.astype(int) 
        
    def _draw_board(self, board_type):

        """
        Use '._draw_root_abstraction' and '._draw_composite' to randomly draw a board with a single root, a pair of roots or a single composite abstraction.

        'board_type' can be 'single', 'double' or 'composition'.
        returns a board and an abstraction mask.

        Args:
            'board_type' (str): type of board to draw. Can be 'single', 'double' or 'composition'.
        Returns:
            'board' (np.array): matrix representing the board.
            'abs_mask' (np.array): matrix indicating which root abstraction is at each position.
        """
        
        assert self.board_dim >= self.abs_dim, "[drawing board] error: board_dim < abs_dim"
        assert self.board_dim >= self.comp_dim * self.abs_dim + (self.comp_dim - 1) * self.comp_margin, "[drawing board] error: board_dim < comp_dim * abs_dim + comp_margin"
        
        board = -1 * np.ones((self.board_dim, self.board_dim)) # give tokens at position
        abs_mask = -1 * np.ones((self.board_dim, self.board_dim)) # indicates which root abstraction covers a given token on the board.

        if board_type == 'single':

            if self.sets is None:
                curr_root = np.random.choice(self.abs_n, (1,)).astype(int).item()
                curr_pos = np.random.choice(self.board_dim - self.abs_dim + 1, (2,)).astype(int)
                curr_bck = np.random.choice(self.vocab_bck, (1,)).astype(int).item()

            else:
                curr_set = self.sets[f'singleRoot_{self.set_type}']
                assert curr_set.shape[0] > 0, "[drawing board] error: no single root abstraction in the current set"

                # randomly sample one row from the set
                rand_row = np.random.choice(curr_set.shape[0], (1,)).astype(int).item()
                curr_root = curr_set[rand_row, 0].astype(int).item()
                curr_pos = curr_set[rand_row, 1:3].astype(int)
                curr_bck = curr_set[rand_row, 3].astype(int).item()

            curr_abs_collapse = self._draw_root_abstraction(curr_root)
                
            i_s = curr_pos[0]
            i_e = i_s + self.abs_dim
            j_s = curr_pos[1]
            j_e = j_s + self.abs_dim
            
            board[i_s:i_e, j_s:j_e] = curr_abs_collapse
            abs_mask[i_s:i_e, j_s:j_e] = curr_root
            board = np.where(board ==  -1, curr_bck, board)


        elif board_type == 'double':

            if self.sets is None:
                curr_root1 = np.random.choice(self.abs_n, (1,)).astype(int).item()
                curr_pos1 = np.random.choice(self.board_dim - self.abs_dim + 1, (2,)).astype(int)
                curr_root2 = np.random.choice(self.abs_n, (1,)).astype(int).item()
                curr_pos2 = np.random.choice(self.board_dim - self.abs_dim + 1, (2,)).astype(int)
                curr_bck = np.random.choice(self.vocab_bck, (1,)).astype(int).item()
            else:
                curr_set = self.sets[f'doubleRoot_{self.set_type}']
                assert curr_set.shape[0] > 0, "[drawing board] error: no double root abstraction in the current set"

                # randomly sample one row from the set
                rand_row = np.random.choice(curr_set.shape[0], (1,)).astype(int).item()
                curr_root1 = curr_set[rand_row, 0].astype(int).item()
                curr_pos1 = curr_set[rand_row, 1:3].astype(int)
                curr_root2 = curr_set[rand_row, 3].astype(int).item()
                curr_pos2 = curr_set[rand_row, 4:6].astype(int)
                curr_bck = curr_set[rand_row, 6].astype(int).item()

            curr_abs_collapse1 = self._draw_root_abstraction(curr_root1)
            
            i_s = curr_pos1[0]
            i_e = i_s + self.abs_dim
            j_s = curr_pos1[1]
            j_e = j_s + self.abs_dim

            board[i_s:i_e, j_s:j_e] = curr_abs_collapse1
            abs_mask[i_s:i_e, j_s:j_e] = curr_root1

            curr_abs_collapse2 = self._draw_root_abstraction(curr_root2)

            i_s = curr_pos2[0]
            i_e = i_s + self.abs_dim
            j_s = curr_pos2[1]
            j_e = j_s + self.abs_dim

            board[i_s:i_e, j_s:j_e] = curr_abs_collapse2
            abs_mask[i_s:i_e, j_s:j_e] = curr_root2

            board = np.where(board ==  -1, curr_bck, board)

        elif board_type == 'composition':
            if self.sets is None:
                curr_comp = np.random.choice(self.comp_n, (1,)).astype(int).item()
                curr_pos = np.random.choice(self.board_dim - (self.comp_dim * self.abs_dim + (self.comp_dim - 1) * self.comp_margin) + 1, (2,)).astype(int)
                curr_bck = np.random.choice(self.vocab_bck, (1,)).astype(int).item()

            else:
                curr_set = self.sets[f'singleComposite_{self.set_type}']
                assert curr_set.shape[0] > 0, "[drawing board] error: no single composite abstraction in the current set"

                # randomly sample one row from the set
                rand_row = np.random.choice(curr_set.shape[0], (1,)).astype(int).item()
                curr_comp = curr_set[rand_row, 0].astype(int).item()
                curr_pos = curr_set[rand_row, 1:3].astype(int)
                curr_bck = curr_set[rand_row, 3].astype(int).item()

            curr_composite, curr_abs_mask = self._draw_composite(self.compositions[curr_comp], self.comp_margin)
            N = curr_composite.shape[0]

            i_s = curr_pos[0]
            i_e = i_s + N
            j_s = curr_pos[1]
            j_e = j_s + N

            board[i_s:i_e, j_s:j_e] = curr_composite
            abs_mask[i_s:i_e, j_s:j_e] = curr_abs_mask
            board = np.where(board ==  -1, curr_bck, board)

        return board.astype(int), abs_mask.astype(int)
        
    def _draw_board_deterministic(self, 
                                  mode = 'root',
                                  root_abstractions = [0],
                                  root_positions = None, 
                                  composite = None,
                                  composite_position = None, 
                                  composite_margin = 0, 
                                  token_bck = None):
        
        """
        Draws a board deterministically.
        
        Args:
            'mode' (str): mode of drawing. Can be 'root' (for single or double) or 'composite'.
            'root_abstractions' (list): list of root abstractions to draw. If entry is -1, a random root abstraction is drawn.
            'root_positions' (np.array): positions of the root abstractions on the board. If None, random positions are used.
            'composite' (int): index of the composite abstraction to draw. If None, random composite abstraction is drawn.
            'composite_position' (np.array): position of the composite abstraction on the board. If None, random position is drawn.
            'composite_margin' (int): margin between constituent root abstractions.
            'token_bck' (int): token to fill empty positions with.
        Returns:
            'board' (np.array): matrix representing the board.
            'abs_mask' (np.array): matrix indicating which root abstraction is at each position.
        """
        
        assert mode in ['root', 'composite'], "please select a recognized mode"

        board = -1 * np.ones((self.board_dim, self.board_dim))
        abs_mask = -1 * np.ones((self.board_dim, self.board_dim))

        if mode == 'root':

            nb_root_abs = len(root_abstractions)

            if root_abstractions[0] == -1:
                root_abstractions = np.random.choice(self.abs_n, (nb_root_abs,)).astype(int)
            else:
                root_abstractions = np.array(root_abstractions).astype(int)

            if root_positions is None:
                root_positions = np.random.choice(self.board_dim - self.abs_dim + 1, (nb_root_abs,2)).astype(int)
            else:
                root_positions = root_positions.astype(int)
            assert len(root_abstractions) == root_positions.shape[0], "nb mismatch"

            for idx_abs in range(nb_root_abs):
                
                curr_abs = self._draw_root_abstraction(root_abstractions[idx_abs])
                
                i_s = root_positions[idx_abs, 0]
                i_e = i_s + self.abs_dim
                j_s = root_positions[idx_abs, 1]
                j_e = j_s + self.abs_dim
                
                board[i_s:i_e, j_s:j_e] = curr_abs
                abs_mask[i_s:i_e, j_s:j_e] = idx_abs

        if mode == 'composite':
            
            if composite is None:
                composite = np.random.choice(self.comp_n, (1,)).astype(int).item()
            assert composite in [i for i in range(self.comp_n)], "composite idx out of range"
            
            curr_composite, curr_abs_mask = self._draw_composite(self.compositions[composite], composite_margin)
            N = curr_composite.shape[0]
            assert N <= self.board_dim, f'error: composite dim ({N}) > board dim ({self.board_dim})'

            if composite_position is None:
                composite_position = np.random.choice(self.board_dim - N + 1, (2,)).astype(int)
            else:
                composite_position = composite_position.astype(int)
            
            i_s = composite_position[0]
            i_e = i_s + N
            j_s = composite_position[1]
            j_e = j_s + N
            board[i_s:i_e, j_s:j_e] = curr_composite
            abs_mask[i_s:i_e, j_s:j_e] = curr_abs_mask
                
        
        # randomly choose the bck token
        if token_bck is None:
            token_bck = np.random.choice(np.arange(self.vocab_bck), (1,)).item()

        assert token_bck in [i for i in range(self.vocab_bck)], "bck token ouutside range"
        board = np.where(board ==  -1, token_bck, board)
                
        return board.astype(int), abs_mask.astype(int)

    def render_board(self, board, board_abs, mask, figsize = (10,10), label_size = (8, 0.2), save_path = None):

        """
        Renders a board and draw it with matplotlib.

        Args:
            'board' (np.array): matrix representing the board.
            'board_abs' (np.array): matrix indicating which root abstraction is at each position.
            'mask' (np.array): matrix indicating which tokens are masked.
            'figsize' (tuple): size of the figure.
            'label_size' (tuple): size of the labels.
            'save_path' (str): path to save the figure. If None, figure is not saved.
        Returns:
            None
        """
            
        h, w = board.shape
        
        # create figure
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(111)
        
        # plot board
        vocab_size = self.vocab_bck + self.vocab_token
        ax.imshow(board, cmap = 'Set3', vmin=0, vmax=vocab_size)

        # add labels
        for k in range(h):
            for l in range(h):
                if board_abs[k,l] >= 0: #if there is an abstraction there
                    ax.text(l-label_size[1], k+label_size[1], str(board[k,l]), size=label_size[0], color = 'darkslategray')            

        # draw baord grid lines
        for j in range(0,h+1):
            rect = Rectangle((j-0.5,-0.5),1,h,linewidth=1,edgecolor='gainsboro',facecolor='none')
            ax.add_patch(rect)

            rect = Rectangle((-0.5,j-0.5),h,1,linewidth=1,edgecolor='gainsboro',facecolor='none')
            ax.add_patch(rect)

        # remove ticks
        ax.tick_params(left = False, right = False , labelleft = False ,
        labelbottom = False, bottom = False)

        # circle masked & add attn
        for k in range(h):
            for l in range(h):
                if mask[k,l] == 1: # token to predict
                    rect = Rectangle((l-0.5,k-0.5),1,1,linewidth=2,edgecolor='none',facecolor='k', alpha = 0.5)
                    ax.add_patch(rect)
        
        # remove axis
        ax.axis('off')

        # save figure
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', pad_inches = 0)

        plt.show()
    
    #//////////////////////////////////////////////////////
    # PYTORCH DATASET FUNCTIONS
    # /////////////////////////////////////////////////////
    
    def __getitem__(self, index, verbose = False):

        """
        Randomly sample a board type (single root, double root or composite) based on 'self.board_types_proba' and returns a randomly drawn board. 

        Args:
            'index' (int): index of the board to draw. Kept for consistency with PyTorch datasets.
            'verbose' (bool): whether to print the board or not.
        Returns:
            'board' (np.array): matrix representing the board.
        """
        
        # ensure proba are proba (sum = 1)
        self.board_types_proba = self.board_types_proba / np.sum(self.board_types_proba)
        
        # choose type
        idx_type = np.random.choice(self.board_types_proba.size, (1,), p = self.board_types_proba).item()
        board_type = self.board_types[idx_type]
        board, _ = self._draw_board(board_type)

        if verbose: print(f'{board_type}:\n{board}')
        
        return board
    
    def __len__(self):
        """
        Returns the length of the dataset. Kept for consistency with PyTorch datasets.

        Args:
            None
        Returns:
            'self.dataset_len' (int): length of the dataset.
        """
        return self.dataset_len
    
    #//////////////////////////////////////////////////////
    # CUSTOM ACCESS FUNCTIONS
    # /////////////////////////////////////////////////////

    def get_n_training_boards(self, n, verbose = False):

        """
        Returns n boards from the training set.

        Args:
            'n' (int): number of boards to draw.
            'verbose' (bool): whether to print the board or not.
        Returns:
            'boards' (np.array): (n, board_dim, board_dim) tensor representing the boards.
            'board_types' (np.array): (n,) Type of each of the n boards.
        """

        set_type_before = self.set_type
        self.set_type = 'train' # set_type to train

        # ensure proba are proba (sum = 1)
        self.board_types_proba = self.board_types_proba / np.sum(self.board_types_proba)

        boards = np.zeros((n, self.board_dim, self.board_dim), dtype = int)
        board_types = np.zeros((n,), dtype = int)

        for i in range(n):
        
            # choose type
            idx_type = np.random.choice(self.board_types_proba.size, (1,), p = self.board_types_proba).item()
            board_type = self.board_types[idx_type]
            board, _ = self._draw_board(board_type)

            if verbose: print(f'{board_type}:\n{board}')

            boards[i] = board
            board_types[i] = idx_type

        # set_type back to before
        self.set_type = set_type_before
        
        return boards.astype(int), board_types.astype(int)

    def get_all_boards(self, target_set = 'test', count_max = None):
        
        """
        Return 'count_max' in the 'target_set'.

        Args:
            'target_set' (str): target set to draw boards from. Can be 'train', 'val' or 'test'.
            'count_max' (int): maximum number of boards to draw. If None, all boards are drawn.
        Returns:
            'boards' (np.array): (n, board_dim, board_dim) tensor representing the boards.
            'board_types' (np.array): (n,) Type of each of the n boards.
        """

        # make single root boards
        curr_set = np.copy(self.sets[f'singleRoot_{target_set}'])
        # shuffle 
        np.random.shuffle(curr_set)
        # select only count_max boards
        if count_max is not None:
            curr_set = curr_set[:min(curr_set.shape[0],count_max)]

        assert curr_set.shape[0] > 0, "[drawing board] error: no single root abstraction in the target set"
        boards_singleRoot = np.zeros((curr_set.shape[0], self.board_dim, self.board_dim))

        for idx_board in range(curr_set.shape[0]):

            curr_root = curr_set[idx_board, 0].astype(int).item()
            curr_pos = curr_set[idx_board, 1:3].astype(int)
            curr_bck = curr_set[idx_board, 3].astype(int).item()

            boards_singleRoot[idx_board], _ = self._draw_board_deterministic(mode = 'root',
                                                                          root_abstractions = [curr_root],
                                                                          root_positions = curr_pos.reshape((1,2)),
                                                                          composite = None,
                                                                          composite_position = None,
                                                                          composite_margin = self.comp_margin,
                                                                          token_bck = curr_bck)

        # make double root boards
        curr_set = np.copy(self.sets[f'doubleRoot_{target_set}'])
        # shuffle 
        np.random.shuffle(curr_set)
        # select only count_max boards
        if count_max is not None:
            curr_set = curr_set[:min(curr_set.shape[0],count_max)]

        assert curr_set.shape[0] > 0, "[drawing board] error: no double root abstraction in the target set"
        boards_doubleRoot = np.zeros((curr_set.shape[0], self.board_dim, self.board_dim))

        for idx_board in range(curr_set.shape[0]):

            curr_root1 = curr_set[idx_board, 0].astype(int).item()
            curr_pos1 = curr_set[idx_board, 1:3].astype(int)
            curr_root2 = curr_set[idx_board, 3].astype(int).item()
            curr_pos2 = curr_set[idx_board, 4:6].astype(int)
            curr_bck = curr_set[idx_board, 6].astype(int).item()

            boards_doubleRoot[idx_board], _ = self._draw_board_deterministic(mode = 'root',
                                                                          root_abstractions = [curr_root1, curr_root2],
                                                                          root_positions = np.concatenate((curr_pos1.reshape((1,2)), curr_pos2.reshape((1,2))), axis = 0),
                                                                          composite = None,
                                                                          composite_position = None,
                                                                          composite_margin = self.comp_margin,
                                                                          token_bck = curr_bck)
                                                                          
        # make composite boards
        curr_set = np.copy(self.sets[f'singleComposite_{target_set}'])
        # shuffle 
        np.random.shuffle(curr_set)
        # select only count_max boards
        if count_max is not None:
            curr_set = curr_set[:min(curr_set.shape[0],count_max)]

        assert curr_set.shape[0] > 0, "[drawing board] error: no composite abstraction in the target set"
        boards_composite = np.zeros((curr_set.shape[0], self.board_dim, self.board_dim))

        for idx_board in range(curr_set.shape[0]):

            curr_comp = curr_set[idx_board, 0].astype(int).item()
            curr_pos = curr_set[idx_board, 1:3].astype(int)
            curr_bck = curr_set[idx_board, 3].astype(int).item()

            boards_composite[idx_board], _ = self._draw_board_deterministic(mode = 'composite',
                                                                         root_abstractions = [],
                                                                         root_positions = None,
                                                                         composite = curr_comp,
                                                                         composite_position = curr_pos,
                                                                         composite_margin = self.comp_margin,
                                                                         token_bck = curr_bck)
            
        # concatenate all boards
        boards = np.concatenate((boards_singleRoot, boards_doubleRoot, boards_composite), axis = 0)
        # create board_types array that indicare the type of each board (0 = single root, 1 = double root, 2 = composite)
        board_types = np.concatenate((np.zeros((boards_singleRoot.shape[0],), dtype = int),
                                        np.ones((boards_doubleRoot.shape[0],), dtype = int),
                                        2 * np.ones((boards_composite.shape[0],), dtype = int)))
        
        return boards.astype(int), board_types.astype(int)
                                                                         
    #//////////////////////////////////////////////////////
    # PRINT INFO ABOUT DATASET
    # /////////////////////////////////////////////////////

    def whoamI(self):

        """
        prints info about the dataset

        Args:
            None
        Returns:
            None
        """
        
        print(f'\n* params:')
        print(f'... board dim:{self.board_dim}')
        print(f'... vocab bck:{self.vocab_bck}')
        print(f'... vocab token:{self.vocab_token}')
        
        print(f'\n* root abstractions ({self.abs_n}):')
        print(f'... dim={self.abs_dim}, c={self.abs_c}, w={self.abs_w_c}, w_m={self.abs_w_m}')
        for idx_abs in range(self.abs_n):
            print(f'... root {idx_abs}:')
            for idx_mode in range(self.abs_w_m):
                print(self.roots[idx_abs,:,:,idx_mode])
                
        print(f'\n* compostions ({self.comp_n}):')
        print(f'... dim={self.comp_dim}, c={self.comp_c}, margin={self.comp_margin}')
        for idx_comp in range(self.comp_n):
            print(f'... comp {idx_comp}:')
            print(self.compositions[idx_comp])

        # print the shapes of the sets
        print(f'\n* sets:')
        if self.sets is not None:
            for set_name in self.sets.keys():
                print(f'... {set_name}: {self.sets[set_name].shape}')
#----------------------------------------------------

#----------------------------------------------------
# test code integrity
#----------------------------------------------------
if __name__ == '__main__':
 
    params_dataset = {
        'dataset_len':1000, #nb of instances in the dataset (only relevant for torch DataLoader)
        'board_dim':8, #dimension of board (height and width). Boards.shape will be (board_dim, board_dim)
        'vocab_bck':10, #nb background tokens
        'vocab_token':10, #nb abstraction tokens (also called object tokens)
        'abs_n': 10, #nb of root abstractions (level #1 objects)
        'abs_dim':3, #footprint (abs_dim, abs_dim) of root abstractions
        'abs_c':9, #cardinality: nb of object tokens in the footprint that will make up the abstraction (abs_c <= abs_dim**2)
        'abs_w_c':None, #nb of fuzzy tokens (use None for deterministic abstractions) amongst the abs_c tokens of the abstraction (abs_w_c <= abs_c) 
        'abs_w_m':1, #nb modes per fuzzy token, default is 1
        'comp_n':5, #nb of composite abstractions (level #2 objects)
        'comp_dim':2, #footprint (comp_dim, comp_dim) of composite abstractions
        'comp_margin':1, #margin between constituent root abstractions
        'comp_c':4, #cardinality: nb of constituent root abstractions in the composite footprint (comp_c <= comp_dim**2)
        'board_types':['single', 'double', 'composition'], #types of boards to generate
        'board_types_proba': np.array([0.33, 0.33, 0.33]), #proba of generating each board type (single, double, composition)
        'run_name': 'test', #name of the run (used for saving)
        'bool_split': True,# whether to split the dataset or not
        'split_method': 'balanced', # method is 'fraction', 'stringent', or 'balanced'
        'card_val':0, # use fraction when split_method is 'fraction', otherwise nb of instances
        'card_test':4, # use fraction when split_method is 'fraction', otherwise nb of instances
    }

    print(f'\nFIRST DATASET')
    # instantiate dataset
    dataset = AbstractionDataset(**params_dataset)
    # get info about dataset
    dataset.whoamI()
    # export dataset
    dataset.export('export/dataset')

    print(f'\nSECOND DATASET')
    # create a new dataset
    dataset = AbstractionDataset(**params_dataset)
    # get info about dataset
    dataset.whoamI()
    # load previously exported dataset
    dataset.load('export/dataset')
    dataset.whoamI()

    # generate K boards by calling the getitem function
    K = 10
    # set the type of set to draw from
    dataset.set_type = 'train'
    # draw K boards
    for idx in range(K):
        print(f'\n* board {idx}:\n{dataset[idx]}')
#----------------------------------------------------