"""
Collection of helper functions
"""

#----------------------------------------------------
# import libraries
#----------------------------------------------------
import os
import shutil
import numpy as np
from PIL import Image
import sys
#----------------------------------------------------

def clear_folder_extension(folder, extension):
    """
    Clear all files with a specific `extension` from `folder`.

    Args:
        folder (str): path to folder
        extension (str): extension of file to be deleted
    Returns:
        None
    """
    for filename in os.listdir(folder):
        if filename.endswith(extension):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

def clear_folder(folder):

    """
    Clear all files from `folder`.

    Args:
        folder (str): path to folder
    Returns:
        None
    """

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def find_best_plot_dim(n):

    """
    Find the best plot dimensions for to organize n plots on a squarish grid.

    Args:
        n (int): number of plots
    Returns:
        (int, int): dimensions of the grid
    """

    dividers = []
    reminders = []
    for i in range(1,n+1):
        if n//i == n/i:
            dividers.append(i)
            reminders.append(n//i)
            
    dividers = np.array(dividers).astype(int)
    reminders = np.array(reminders).astype(int)
    s = np.absolute(dividers - reminders)
    i = np.where(s == s.min())[0][0]
    
    return (dividers[i], reminders[i])

def get_ij(x, q):

    """
    Get the index of a query q in a matrix x.

    Args:
        x (np.array): matrix
        q (int): query
    Returns:
        (int, int): index of q in x
    """

    ci,cj = np.where(x == q)
    i,j = ci[0], cj[0]
    return (i,j)

def get_concat_h(images, margin = 0):

    """
    Concatenate images horizontally

    Args:
        images (list): list of images
        margin (int): margin between images
    Returns:
        dst (Image): concatenated image
    """
    
    # compute total width
    w, h = 0, 0
    for i, img in enumerate(images):
        w += img.width
        h = img.height
    w += (len(images) - 1) * margin
        
    dst = Image.new('RGB', (w, h))
    
    w_counter = 0
    for i, img in enumerate(images):
        dst.paste(img, (w_counter, 0))
        w_counter += img.width + margin
        
    return dst

def get_concat_v(images, margin):

    """
    Concatenate images vertically

    Args:
        images (list): list of images
        margin (int): margin between images
    Returns:
        dst (Image): concatenated image
    """
    
    # compute total height
    w, h = 0, 0
    for i, img in enumerate(images):
        w = img.width 
        h += img.height 
    h += (len(images) - 1) * margin
        
    dst = Image.new('RGB', (w, h))
    
    h_counter = 0
    for i, img in enumerate(images):
        dst.paste(img, (0, h_counter))
        h_counter += img.height + margin
        
    return dst

def merge_attn_plots(path, heads, blocks, prefix = 'attn'):

    """
    Merge attention plots

    Args:
        path (str): path to plots
        heads (list): list of heads
        blocks (list): list of blocks
        prefix (str): prefix of the plots
    """

    scaling, margin = 1, 10

    # load all images
    images = []
    for idx_block, block in enumerate(blocks):
        images_heads = []
        
        for idx_head, head in enumerate(heads):
            img_path = path + f'{prefix}_b{block}_h{head}.jpg'
            print(img_path)
            img = Image.open(img_path)
            img = img.resize((img.width//scaling, img.height//scaling))
            images_heads.append(img)
            
        img_comp = get_concat_v(images_heads, margin)
        images.append(img_comp)
        
    img_comp = get_concat_h(images, margin)
            
    path_save = path + f'{prefix}_merge.jpg'
    img_comp.save(path_save)

def divide_into_batches(lst, n):
    """
    Divide a list into n batches of similar size.

    Args:
    - lst (list): The list to be divided.
    - n (int): The number of batches.

    Returns:
    - batches: List of lists, each inner list is a batch.
    """
    
    # Length of the list
    N = len(lst)
    
    # Basic size of each batch
    size = N // n
    
    # Any remainders
    remainder = N % n
    
    batches = []
    start = 0
    
    for i in range(n):
        # For the first 'remainder' batches, add an extra item
        if remainder > 0:
            batches.append(lst[start:start+size+1])
            start += size + 1
            remainder -= 1
        else:
            batches.append(lst[start:start+size])
            start += size

    return batches

def progress_bar(iterable, bar_length=50):

    """
    Progress bar
    
    Args:
        iterable (iterable): iterable object
        bar_length (int): length of the progress bar
    Returns:
        None
    """

    total = len(iterable)
    for i, item in enumerate(iterable):
        percent = (i + 1) / total
        arrow = "=" * int(round(percent * bar_length) - 1)
        spaces = " " * (bar_length - len(arrow))

        sys.stdout.write(f"\r[{arrow + spaces}] {int(percent * 100)}%")
        sys.stdout.flush()

        yield item

def print_dict(d, indent=0):
    """
    Print the tree of keys and values types of a dictionary recursively.

    Args:
        d (dict): dictionary
        indent (int): indentation level
    Returns:
        None
    """

    for k, v in d.items():
        if isinstance(v, dict):
            print('...' * indent + str(k) + ':')
            print_dict(v, indent+1)
        else:
            print('...' * indent + str(k) + ':' + str(type(v)))
            # if type is np.array, print shape
            if isinstance(v, np.ndarray):
                print('...' * (indent+1) + 'shape: ' + str(v.shape))