import torch
from torch.distributions import Categorical
from ortools_tsp import my_solve


class Pi2AP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pi):
        dist = Categorical(pi.transpose(2, 1))
        action_int = dist.sample()
        log_prob = dist.log_prob(action_int)
        action = torch.tensor(action_int, dtype=torch.float32)
        return action, log_prob

    @staticmethod
    def backward(ctx, dcda, dldp):
