import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):

    """
    An implementation of GRU.
    from: https://github.com/emadRad/lstm-gru-pytorch
    """

    def __init__(self, dim_input, dim_hidden, dim_output, bias=True):
        super(GRU, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.bias = bias
        self.x2h = nn.Linear(dim_input, 3 * dim_hidden, bias=bias)
        self.h2h = nn.Linear(dim_hidden, 3 * dim_hidden, bias=bias)
        self.h2o = nn.Linear(dim_hidden, dim_output, bias=bias)
        self.init_weight()
        self.zero_init_state = False


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
        # reformat x
        # x = x.view(-1, x.size(1))
        hidden = hidden.view(-1)
        # compute the gates
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)
        # reformat
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        # sep data
        i_r, i_i, i_n = gate_x.chunk(3, 0)
        h_r, h_i, h_n = gate_h.chunk(3, 0)
        # compute the 2 gates
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))
        # update hidden
        h = newgate + inputgate * (hidden - newgate)
        y = self.h2o(h)

        output = [y, h]
        cache = [resetgate, inputgate, newgate]
        return output, cache

    def get_init_states(self):
        if self.zero_init_state:
            return self.get_zero_states()
        return self.get_rand_states()

    # @torch.no_grad()
    def get_zero_states(self):
        h_0 = torch.zeros(1, 1, self.dim_hidden)
        return h_0

    # @torch.no_grad()
    def get_rand_states(self, scale=.1):
        h_0 = torch.randn(1, 1, self.dim_hidden) * scale
        return h_0


if __name__ == '__main__':
    '''how to use '''
    print(torch.__version__)
    dim_input = 5
    dim_hidden = 6
    dim_output = 7
    agent = GRU(dim_input=dim_input, dim_hidden=dim_hidden, dim_output=dim_output)

    h = agent.get_init_states()
    x = torch.rand((dim_input, ))

    [yhat, h], cache = agent.forward(x, h)
    print(h)
    print(yhat)
