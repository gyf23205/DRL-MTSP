import torch


class N():
    def __init__(self):
        self.c = torch.tensor([[3., 2.]], requires_grad=True)
        self.d = torch.tensor([[1., 2.]], requires_grad=True)

    def __call__(self, x):
        return torch.mm(x, self.c) + self.d

    def parameters(self):
        return [self.c, self.d]


class S():
    def __init__(self):
        self.a = torch.tensor([[3.], [2.]], requires_grad=True)
        self.b = torch.tensor([1.], requires_grad=True)

    def __call__(self, prob):
        return torch.mm(prob, self.a) + self.b

    def parameters(self):
        return [self.a, self.b]


def get_cost(prob):
    cost = float(2 * prob.sum())
    return cost

#
# def surrogate(prob):
#     a = torch.tensor([2], requires_grad=True)
#     b = torch.tensor([1], requires_grad=True)
#     return torch.mul(prob.sum(), a) + b


# device = 'cuda'
# batch = 10
n_node = 2
x = torch.tensor([[2.]], requires_grad=False)
policy = N()
surrogate = S()
prob = policy(x)
cost = get_cost(prob)
# estimate cost via the surrogate network
cost_s = surrogate(prob)
# compute loss, need to freeze surrogate's parameters
loss = torch.mul(torch.tensor(cost), prob.sum(dim=1)).sum() \
       - torch.mul(cost_s.detach(), prob.sum(dim=1)).sum() \
       + (cost_s).sum()
print(loss)
# cost_s.detach() in 39
# compute gradient's variance loss w.r.t. surrogate's parameter
grad_p = torch.autograd.grad(loss, policy.parameters(),
                             grad_outputs=torch.ones_like(loss), create_graph=True, retain_graph=True)
grad_p = torch.cat([torch.reshape(p, [-1]) for p in grad_p], 0)
grad_ps = torch.square(grad_p).mean(0)
grad_s = torch.autograd.grad(grad_ps, surrogate.parameters(),
                             grad_outputs=torch.ones_like(grad_ps), retain_graph=True)
print()
