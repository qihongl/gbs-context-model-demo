import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleContextNetwork(nn.Module):
    """a context MLP network

              output
                |ho
         elementwise multiply
               /   \
             h       cr
             |ih     |cih
             x       c_input

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

    def __init__(self, dim_input, dim_hidden, dim_output, dim_cinput, w_lr=1e-4, c_lr=1e-3, seed=0):
        '''
        w_lr: lr for the regular weights
        c_lr: lr for the context rep
        '''
        super(SimpleContextNetwork, self).__init__()
        torch.manual_seed(seed)
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.dim_cinput = dim_cinput
        # weights
        self.ih = nn.Linear(dim_input, dim_hidden)
        self.cih = nn.Linear(dim_cinput, dim_hidden)
        self.ho = nn.Linear(dim_hidden, dim_output)
        #
        self.w_lr = w_lr
        self.c_lr = c_lr
        #
        self.init_context_rep()
        self.unfreeze_regular_wts()
        init_weight(self)
        # opts
        regular_wts = [{'params': self.ih.parameters()}, {'params': self.ho.parameters()}]
        context_wts = [{'params': self.cih.parameters()}]
        self.w_optimizer = torch.optim.Adam(regular_wts, lr=w_lr)
        self.c_optimizer = torch.optim.Adam(context_wts, lr=c_lr)
        self.criterion = nn.MSELoss()


    def init_context_rep(self):
        # self.context_rep = torch.rand(size=(self.dim_cinput, ))
        self.context_rep = torch.ones(size=(self.dim_cinput, ))


    def forward(self, x, return_act=False):
        # forward
        h = F.relu(self.ih(x))
        cr = self.cih(self.context_rep).sigmoid()
        y = self.ho(cr * h).sigmoid()
        #
        if return_act:
            return y, [cr, h]
        return y


    @torch.no_grad()
    def get_context_rep(self, x, return_all=False):
        _, [cr, h] = self.forward(x=x, return_act=True)
        if return_all:
            return [cr, h]
        return cr


    def update_context(self, y, x, n_iters=10, threshold=1e-2, verbose=True):
        '''
        x (n x feature_dim) or (feature_dim, )
        y (n x 1)           or (1, )
        ... where n is the batch size

        '''
        self.freeze_regular_wts()
        # find context rep
        for i in range(n_iters):
            loss = self.criterion(self.forward(x), y)
            # w update
            self.c_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.c_optimizer.step()
        self.unfreeze_regular_wts()
        return self.get_context_rep(x)


    def update_weights(self, loss):
        self.w_optimizer.zero_grad()
        loss.backward()
        self.w_optimizer.step()


    def unfreeze_regular_wts(self):
        'OPPOSITE for the context layer'
        self.ih.weight.requires_grad = True
        self.ih.bias.requires_grad = True
        self.ho.weight.requires_grad = True
        self.ho.bias.requires_grad = True
        # context rep
        self.cih.weight.requires_grad = False
        self.cih.bias.requires_grad = False


    def freeze_regular_wts(self):
        'OPPOSITE for the context layer'
        self.ih.weight.requires_grad = False
        self.ih.bias.requires_grad = False
        self.ho.weight.requires_grad = False
        self.ho.bias.requires_grad = False
        # context rep
        self.cih.weight.requires_grad = True
        self.cih.bias.requires_grad = True


    def freeze_all(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad = False
            # print(name)


    def unfreeze_all(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad = True


def init_weight(agent, option = 'ortho'):
    print(f'weight init = {option}')
    for name, parameter in agent.named_parameters():
        if 'weight' in name:
            if option == 'ortho':
                nn.init.orthogonal_(parameter)
            else:
                nn.init.xavier_uniform_(parameter)
        elif 'bias' in name:
            nn.init.constant_(parameter, .1)
        print(name)
        print(parameter)


if __name__ == '__main__':
    '''how to use '''

    dim_input, dim_hidden, dim_output, dim_cinput = 3,4,1,6
    agent = SimpleContextNetwork(dim_input, dim_hidden, dim_output, dim_cinput)


    batch_size = 10
    x = torch.rand((batch_size, dim_input))
    yhat = agent.forward(x)
    print(yhat)


    y = torch.ones((batch_size, 1))

    agent.criterion(y, yhat)

    context_rep = agent.update_context(y, x)

    init_weight(agent)
