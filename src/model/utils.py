import torch
import numpy as np
from torch.nn.functional import smooth_l1_loss


'''helpers'''

eps = np.finfo(np.float32).eps.item()


def compute_returns(rewards, gamma=0, normalize=False):
    """compute return in the standard policy gradient setting.

    Parameters
    ----------
    rewards : list, 1d array
        immediate reward at time t, for all t
    gamma : float, [0,1]
        temporal discount factor
    normalize : bool
        whether to normalize the return
        - default to false, because we care about absolute scales

    Returns
    -------
    1d torch.tensor
        the sequence of cumulative return

    """
    R = 0
    returns = []
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + eps)
    return returns


def get_reward(a_t, a_t_targ):
    """define the reward function at time t

    Parameters
    ----------
    a_t : int
        action
    a_t_targ : int
        target action

    Returns
    -------
    torch.FloatTensor, scalar
        immediate reward at time t

    """
    if a_t == a_t_targ:
        r_t = 1
    else:
        r_t = 0
    return torch.tensor(r_t).type(torch.FloatTensor).data


def compute_a2c_loss(probs, values, returns):
    """compute the objective node for policy/value networks

    Parameters
    ----------
    probs : list
        action prob at time t
    values : list
        state value at time t
    returns : list
        return at time t

    Returns
    -------
    torch.tensor, torch.tensor
        Description of returned object.

    """
    policy_grads, value_losses = [], []
    for prob_t, v_t, R_t in zip(probs, values, returns):
        A_t = R_t - v_t.item()
        policy_grads.append(-prob_t * A_t)
        value_losses.append(
            smooth_l1_loss(torch.squeeze(v_t), torch.squeeze(R_t))
        )
    loss_policy = torch.stack(policy_grads).sum()
    loss_value = torch.stack(value_losses).sum()
    return loss_policy, loss_value


def sparse_with_mean(tensor, sparsity, mean= 1.,  std=0.01):
    r"""Coppied from PyTorch source code. Fills the 2D input `Tensor` as a sparse matrix, where the
    non-zero elements will be drawn from the normal distribution
    :math:`\mathcal{N}(0, 0.01)`, as described in `Deep learning via
    Hessian-free optimization` - Martens, J. (2010).

    Args:
        tensor: an n-dimensional `torch.Tensor`
        sparsity: The fraction of elements in each column to be set to zero
        std: the standard deviation of the normal distribution used to generate
            the non-zero values

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.sparse_(w, sparsity=0.1)
    """
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    rows, cols = tensor.shape
    num_zeros = int(math.ceil(sparsity * rows))

    with torch.no_grad():
        tensor.normal_(mean, std)
        for col_idx in range(cols):
            row_indices = torch.randperm(rows)
            zero_indices = row_indices[:num_zeros]
            tensor[zero_indices, col_idx] = 0
    return tensor


def loss_stopped(losses, threshold=1e-6, num_epochs=10):
    """
    Check if the loss has stopped changing by a certain threshold over a specified number of epochs.

    Args:
        losses (list): List of MSE losses over epochs.
        threshold (float): The threshold for change in loss. Defaults to 1e-6.
        num_epochs (int): Number of recent epochs to consider for checking. Defaults to 10.

    Returns:
        bool: True if the loss has stopped changing, False otherwise.
    """
    if len(losses) < num_epochs:
        return False

    recent_losses = losses[-num_epochs:]
    max_change = max(recent_losses) - min(recent_losses)

    if max_change <= threshold:
        return True
    else:
        return False
