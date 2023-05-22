from collections import defaultdict
import torch

class GradientStatistics:
    state: dict
    def __init__(self, param_groups, n_sigmas_thr):
        self.param_groups = param_groups
        self.state = defaultdict(dict)
        self.k_thr = 1/(n_sigmas_thr*n_sigmas_thr)
        self.hist = None
        self.params_cnt = 0

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad
                    state = self.state[p]
                    state['prev_grad'] = torch.zeros_like(grad)

    def step(self):
        """
        Update statistics after backward() call
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad.detach().clone()
                    state = self.state[p]
                    if 'prev_grad' not in state:
                        state['prev_grad'] = torch.zeros_like(grad)
                    if 'n' not in state:
                        state['n'] = torch.zeros_like(grad).double()
                        state['n_pos'] = torch.zeros_like(grad).double()
                        state['sum'] = torch.zeros_like(grad).double()
                        state['sum_sqr'] = torch.zeros_like(grad).double()
                        state['trust_count'] = torch.zeros_like(grad)
                    grad_delta = grad - state['prev_grad']
                    state['n'] += 1
                    # state['n_pos'] += (grad.sign() + 1)*0.5
                    state['sum'] += grad_delta
                    state['sum_sqr'] += grad_delta.square()
                    state['prev_grad'] = grad

    def reset_stat(self):
        self.hist = None
        self.params_cnt = 0

    def update_stat(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'n' in state:
                    self.params_cnt += p.numel()
                    k = state['sum_sqr']/(state['n']*state['sum'].square()) - 1
                    trusted_gradients_mask = (state['n'] > 5) & (k < self.k_thr)
                    hist = torch.histogram( (state['n'][trusted_gradients_mask]).log().cpu(), 20, range=(0,20))
                    if self.hist is None:
                        self.hist = hist[0]
                    else:
                        self.hist += hist[0]
                    state['trust_count'][trusted_gradients_mask] += 1
                    state['n'][trusted_gradients_mask] = 0
                    # state['n_pos']
                    state['sum'][trusted_gradients_mask] = 0
                    state['sum_sqr'][trusted_gradients_mask] = 0

    def reset_trust_count(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'n' in state:
                    state['trust_count'].zero_()

    def get_trust_hist(self):
        trust_hist = None
        params_cnt = 0
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'n' in state:
                    params_cnt += p.numel()
                    # hist = torch.histogram( (state['trust_count']+1).log().cpu(), 20, range=(0,20))
                    hist = torch.histogram( state['trust_count'].cpu(), 40, range=(0,40))
                    if trust_hist is None:
                        trust_hist = hist[0]
                    else:
                        trust_hist += hist[0]
        return trust_hist





