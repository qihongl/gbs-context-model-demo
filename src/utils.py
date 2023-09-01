import os
import sys
import warnings
import torch
import pickle
import numpy as np
from itertools import product



def to_pth(np_array, pth_dtype=torch.FloatTensor):
    return torch.tensor(np_array).type(pth_dtype)


def to_sqpth(np_array, pth_dtype=torch.FloatTensor):
    return torch.squeeze(to_pth(np_array, pth_dtype=pth_dtype))


def to_np(torch_tensor):
    return torch_tensor.data.numpy()


def to_sqnp(torch_tensor, dtype=np.float64):
    return np.array(np.squeeze(to_np(torch_tensor)), dtype=dtype)


def enumerated_product(*args):
    # https://stackoverflow.com/questions/56430745/enumerating-a-tuple-of-indices-with-itertools-product
    yield from zip(product(*(range(len(x)) for x in args)), product(*args))


def ignore_warnings():
    if not sys.warnoptions:
        warnings.simplefilter("ignore")


def pickle_save_dict(input_dict, save_path):
    """Save the dictionary

    Parameters
    ----------
    input_dict : type
        Description of parameter `input_dict`.
    save_path : type
        Description of parameter `save_path`.

    """
    with open(save_path, 'wb') as handle:
        pickle.dump(input_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load_dict(fpath):
    """load the dict

    Parameters
    ----------
    fpath : type
        Description of parameter `fpath`.

    Returns
    -------
    type
        Description of returned object.

    """
    return pickle.load(open(fpath, "rb"))


def random_walk(n, step_size=.1):
    x = 0
    X = np.zeros(n)
    for t in range(n):
        dx = np.random.choice([step_size, -step_size])
        x += dx
        X[t] = x
    return X

def binarize(x):
    x[x>0] = 1
    x[x<=0] = 0
    return x


def mkdir(dir_name, verbose=False):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        if verbose:
            print(f'Dir created: {dir_name}')
    else:
        if verbose:
            print(f'Dir exist: {dir_name}')
