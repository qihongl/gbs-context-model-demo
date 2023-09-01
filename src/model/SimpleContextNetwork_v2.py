import torch
import torch.nn as nn
from model.utils import loss_stopped
# from torch.optim.lr_scheduler import LinearLR
# from model.utils import sparse_with_mean

class SimpleContextNetwork_v2(nn.Module):
    """a context MLP network

              output
                |ho
         elementwise multiply
               /   \
             h       cr
             |ih
             x

    Parameters
    ----------
    dim_input : int
        dim state space
    dim_hidden : int
        # hidden units
    dim_output : int
        dim output space
    dim_c*: int
        dim conext rep
    """

    def __init__(self, dim_input, dim_hidden, dim_output, dim_cinput,
                lr=1e-4, w_lr=1e-4, c_lr=1e-3, seed=0):
        '''
        w_lr: lr for the regular weights
        c_lr: lr for the context rep
        '''
        super(SimpleContextNetwork_v2, self).__init__()
        torch.manual_seed(seed)
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.dim_cinput = dim_cinput
        # weights
        self.ih = nn.Linear(dim_input, dim_hidden)
        self.ho = nn.Linear(dim_hidden, dim_output)
        #
        self.lr = lr
        self.w_lr = w_lr
        self.c_lr = c_lr
        #
        self.c_input_ = torch.relu(torch.rand((dim_cinput, )))
        self.register_parameter(name='c_input',
            param=torch.nn.Parameter(self.c_input_))

        init_weight(self)
        # opts
        self.regular_wts = [{'params': self.ih.parameters()}, {'params': self.ho.parameters()}]
        self.context_wts = [{'params': self.c_input}]
        self.w_optimizer = torch.optim.Adam(self.regular_wts, lr=w_lr)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # log_str
        log_str = f'dim-c-{dim_cinput}-h-{dim_hidden}-o-{dim_output}'
        log_str += f'/lr-{lr}-w_lr-{w_lr}-c_lr-{c_lr}'
        self.log_str = log_str

    def set_context(self, c):
        assert len(c) == self.dim_cinput
        assert torch.is_tensor(c)
        self.register_parameter(name='c_input', param=torch.nn.Parameter(c))
        self.context_wts = [{'params': self.c_input}]

    def forward(self, x, return_act=False):
        # forward
        h = torch.relu(self.ih(x))
        cr = self.c_input.tanh()
        y = self.ho(cr * h).tanh()
        if return_act:
            return y, [self.c_input, cr, h]
        return y


    @torch.no_grad()
    def get_context_rep(self, x, return_all=False):
        _, [c_input, cr, h] = self.forward(x=x, return_act=True)
        if return_all:
            return [c_input, cr, h]
        return c_input


    def update_context(self, y, x, n_iters=100, threshold=1e-2,
                            freeze_weights=True, verbose=True):
        '''
        x (n x feature_dim) or (feature_dim, )
        y (n x 1)           or (1, )
        ... where n is the batch size

        '''
        if n_iters == 0:
            return None
        # find context rep
        losses = []
        self.c_optimizer = torch.optim.Adam(self.context_wts, lr=self.c_lr)
        for i in range(n_iters):
            loss = self.criterion(self.forward(x), y)
            self._update_context(loss)
            #
            losses.append(loss.data)
            loss_stopped_ = loss_stopped(losses, threshold=threshold)
            if loss_stopped_:
                # print(f'loss stopped decreasing at {i}-th iter')
                break
        # if not loss_stopped_:
        #     print(f'loss did NOT converge at {i}-th iter')
        return self.get_context_rep(x), i

    def _update_context(self, loss):
        self.c_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.c_optimizer.step()


    def update_weights(self, loss):
        self.w_optimizer.zero_grad()
        loss.backward()
        self.w_optimizer.step()


def init_weight(agent, option = 'ortho'):
    # print(f'weight init = {option}')
    for name, parameter in agent.named_parameters():
        if 'weight' in name:
            if option == 'ortho':
                nn.init.orthogonal_(parameter)
            else:
                nn.init.xavier_uniform_(parameter)
        elif 'bias' in name:
            nn.init.constant_(parameter, .1)
        # print(name)
        # print(parameter)


if __name__ == '__main__':
    '''how to use '''
    print(torch.__version__)

    from copy import deepcopy
    dim_input, dim_hidden, dim_output, dim_cinput = 6,6,1,6
    agent = SimpleContextNetwork_v2(dim_input, dim_hidden, dim_output, dim_cinput)
    agent.log_str

    batch_size = 10
    x = torch.rand((batch_size, dim_input))

    # predict and get loss
    yhat = agent.forward(x)
    y = torch.ones((batch_size, 1))
    loss = agent.criterion(y, yhat)

    # make sure c_input stays the same under weight update
    ih_weight_0 = deepcopy(agent.ih.weight.data)
    c_input_0 = deepcopy(agent.c_input.data)
    agent.update_weights(loss)
    assert torch.any(ih_weight_0 != agent.ih.weight.data), 'ERROR: ih weight did not change!'
    assert all(c_input_0 == agent.c_input.data), 'ERROR: c input changed!'

    # make weights stay the same under context update; and c input should change
    ih_weight_0 = deepcopy(agent.ih.weight.data)
    c_input_0 = deepcopy(agent.c_input.data)
    cr = agent.update_context(y, x)
    assert torch.all(ih_weight_0 == agent.ih.weight.data), 'ERROR: ih weight changed!'
    assert all(c_input_0 != agent.c_input.data), 'ERROR: c input did not change!'
