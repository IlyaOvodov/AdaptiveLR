'''
learning nearning rate
'''
import types
import math
import torch


class LLR_SGD(torch.optim.SGD):
    def __init__(self, *args, lrlr=0, update_mode = 'const', use='grad', **kwargs):
        super(LLR_SGD, self).__init__(*args, **kwargs)
        self.lrlr = lrlr
        self.update_mode = update_mode
        self.use = use

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            sum_sim = 0
            sum_rev = 0

            sum_lr_grad = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                param_state = self.state[p]
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-group['lr'], d_p)
                if self.lrlr:
                    grad = p.grad.data if self.use == 'grad' else d_p
                    if 'lr_grad' in param_state:
                        mul = grad*param_state['lr_grad']
                        sum_lr_grad += mul.sum()
                        #sum_sim += torch.sum(mul>0)
                        #sum_rev += torch.sum(mul < 0)
                    param_state['lr_grad'] = torch.clone(grad).detach()
            if sum_lr_grad:
                if self.update_mode == 'loggrad':
                    #print(group['lr'], sum_lr_grad)
                    group['lr'] = torch.exp( math.log(group['lr']) + self.lrlr*sum_lr_grad).item()
                elif self.update_mode == 'const':
                    if sum_lr_grad > 0:
                        group['lr'] *= (1+self.lrlr)
                    elif sum_lr_grad < 0:
                        group['lr'] /= (1+self.lrlr)
                else:
                    raise Exception("incorrect update_mode "+str(self.update_mode))
        return loss



def _step(self, closure=None):
    """Performs a single optimization step.

    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
        loss = closure()

    for group in self.param_groups:
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        for p in group['params']:
            if p.grad is None:
                continue
            d_p = p.grad.data
            if weight_decay != 0:
                d_p.add_(weight_decay, p.data)
            if momentum != 0:
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf

            p.data.add_(-group['lr'] * d_p)

    return loss


def enable_LLR(trainer, optim, lrlr, device):
    assert isinstance(optim, torch.optim.SGD)
    if lrlr:
        lrs = []
        for group in optim.param_groups:
            lr_param = torch.nn.Parameter(torch.Tensor([group['lr']]).to(device))
            group['lr'] = lr_param
            lrs.append(lr_param)
        lrs_param_groups = {'params': lrs, 'lr': lrlr}
        optim.add_param_group(lrs_param_groups)
        optim.step = types.MethodType(_step, optim)
    return optim
