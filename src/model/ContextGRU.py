import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.utils import loss_stopped

class ContextGRU(nn.Module):

    """
    An implementation of ContextGRU.
    """

    def __init__(self, dim_input, dim_hidden, dim_output, dim_context,
                lr=1e-3, w_lr=1e-3, c_lr=1e-3, threshold=1e-2, n_iters=0, bias=True):
        super(ContextGRU, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.dim_context = dim_context
        self.bias = bias
        self.lr = lr
        self.c_lr = c_lr
        self.w_lr = w_lr
        self.threshold = threshold
        self.n_iters = n_iters
        # misc params
        self.zero_init_state = False
        # init param
        self.x2h = nn.Linear(dim_input, 3 * dim_hidden, bias=bias)
        self.h2h = nn.Linear(dim_hidden, 3 * dim_hidden, bias=bias)
        self.h2o = nn.Linear(dim_hidden, dim_output, bias=bias)
        # init context param
        self.context_ = torch.relu(torch.rand((dim_context, )))
        self.register_parameter(name='context', param=torch.nn.Parameter(self.context_))
        # group the params by regular param vs. context params
        self.context_wts = [{'params': self.context}]
        self.regular_wts = [{'params': self.x2h.parameters()},
                            {'params': self.h2h.parameters()},
                            {'params': self.h2o.parameters()}]
        # set up optimizers
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.w_optimizer = torch.optim.Adam(self.regular_wts, lr=w_lr)
        self.wlr_scheduler = ReduceLROnPlateau(self.w_optimizer, 'min')
        self.criterion = nn.MSELoss()
        # reset weights
        self.init_weight()
        # log_str
        log_str = f'dim-c-{dim_context}-h-{dim_hidden}-o-{dim_output}'
        log_str += f'/lr-{lr}-w_lr-{w_lr}-c_lr-{c_lr}'
        log_str += f'/n_iters-{n_iters}-threshold-{threshold}'
        self.log_str = log_str


    def init_weight(self, option = 'ortho'):
        # print(f'weight init = {option}')
        for name, parameter in self.named_parameters():
            if 'weight' in name:
                if option == 'ortho':
                    nn.init.orthogonal_(parameter)
                else:
                    nn.init.xavier_uniform_(parameter)
            elif 'bias' in name:
                nn.init.constant_(parameter, .1)


    def forward(self, x, hidden):
        # compute the gates
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)
        # # reformat
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        # sep data
        i_r, i_i, i_n = gate_x.chunk(3, 0)
        h_r, h_i, h_n = gate_h.chunk(3, 0)
        # compute the 2 gates
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        # update hidden
        h = newgate + inputgate * (hidden - newgate)
        # compute context representation
        cr = torch.tanh(self.context)
        # cr = torch.sigmoid(self.context)
        # compute output
        y = self.h2o(cr * h)
        # package the output
        output = [y, h]
        cache = [self.context, cr, resetgate, inputgate, newgate]
        return output, cache


    @torch.no_grad()
    def get_context_rep(self, x, hidden):
        [y, h], [context, _, _, _, _] = self.forward(x=x, hidden=hidden)
        return context

    def update_weights(self, loss):
        self.w_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.w_optimizer.step()

    def set_context(self, c):
        assert len(c) == self.dim_context
        assert torch.is_tensor(c)
        self.register_parameter(name='context', param=torch.nn.Parameter(c))
        self.context_wts = [{'params': self.context}]

    def update_context(self, y, x, h0=None, verbose=True):
        '''
        x (n x feature_dim) or (feature_dim, )
        y (n x 1)           or (1, )
        ... where n is the batch size
        '''
        if self.n_iters == 0:
            return None
        # find context rep
        losses = []
        self.c_optimizer = torch.optim.Adam(self.context_wts, lr=self.c_lr)
        for i in range(self.n_iters):
            T = len(y)
            h = self.get_init_states() if h0 is None else h0
            loss = 0
            for t in range(T):
                [yhat, h], _ = self.forward(x[t], h)
                loss += self.criterion(y[t], yhat)
            # update context
            self._update_context(loss)
            #
            losses.append(loss.data)
            loss_stopped_ = loss_stopped(losses, threshold=self.threshold)
            if loss_stopped_:
                # print(f'loss stopped decreasing at {i}-th iter')
                break
        # if not loss_stopped_:
        #     print(f'loss did NOT converge at {i}-th iter')
        return self.context, h, i

    def _update_context(self, loss):
        self.c_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.c_optimizer.step()


    def get_init_states(self):
        if self.zero_init_state:
            return self.get_zero_states()
        return self.get_rand_states()

    # @torch.no_grad()
    def get_zero_states(self):
        # h_0 = torch.zeros(1, 1, self.dim_hidden)
        h_0 = torch.zeros(self.dim_hidden, )
        return h_0

    # @torch.no_grad()
    def get_rand_states(self, scale=.1):
        # h_0 = torch.randn(1, 1, self.dim_hidden) * scale
        h_0 = torch.randn(self.dim_hidden, ) * scale
        return h_0


if __name__ == '__main__':
    '''how to use '''
    print(torch.__version__)

    '''init model'''
    dim_input = 5
    dim_hidden = 6
    dim_output = 7
    dim_context = 6
    lr = w_lr = c_lr = 1e-3
    agent = ContextGRU(dim_input=dim_input, dim_hidden=dim_hidden,
                       dim_output=dim_output, dim_context=dim_context,
                       lr=lr, w_lr=w_lr, c_lr=c_lr)

    '''forward pass'''
    h = agent.get_init_states()
    x = torch.rand((dim_input, ))
    y = torch.ones((dim_output))

    [yhat, h], cache = agent.forward(x, h)
    loss = agent.criterion(y, yhat)

    '''update regular weights'''
    agent.update_weights(loss)

    '''update context'''
    h = agent.get_init_states()
    x = torch.rand((dim_input, ))
    y = torch.ones((dim_output))

    agent.update_context(y, x)

    '''get context rep'''
    cr = agent.get_context_rep(x, h)
    print(cr)
