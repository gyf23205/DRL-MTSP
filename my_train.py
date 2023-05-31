from policy import Policy, action_sample, get_cost, Surrogate
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from validation import validate
import numpy as np


def train(batch_size, no_nodes, policy_net, surrogate, lr_p, lr_s, no_agent, iterations, device):

    # prepare validation data
    validation_data = torch.load('./validation_data/validation_data_'+str(no_nodes)+'_'+str(batch_size))
    # a large start point
    best_so_far = np.inf
    validation_results = []

    # optimizer
    optim_p = torch.optim.Adam(policy_net.parameters(), lr=lr_p)
    optim_s = torch.optim.Adam(surrogate.parameters(), lr=lr_s)

    for itr in range(iterations):
        # prepare training data
        data = torch.load('./training_data/training_data_'+str(no_nodes)+'_'+str(batch_size))  # [batch, nodes, fea], fea is 2D location
        adj = torch.ones([data.shape[0], data.shape[1], data.shape[1]])  # adjacent matrix fully connected
        data_list = [Data(x=data[i], edge_index=torch.nonzero(adj[i], as_tuple=False).t()) for i in range(data.shape[0])]
        batch_graph = Batch.from_data_list(data_list=data_list).to(device)

        # get pi
        pi = policy_net(batch_graph, n_nodes=data.shape[1], n_batch=batch_size)
        # sample action and calculate log probabilities
        action, log_prob = action_sample(pi)
        # get real cost for each batch
        cost = get_cost(action, data, no_agent)  # cost: tensor [batorch.cat([torch.reshape(p, [-1]) for p in pg_grads], 0)tch, 1]
        # estimate cost via the surrogate network
        cost_s = torch.squeeze(surrogate(log_prob))
        # compute loss, need to freeze surrogate's parameters
        loss = torch.mul(torch.tensor(cost, device=device) - 2, log_prob.sum(dim=1)).sum() \
               - torch.mul(cost_s.detach() - 2, log_prob.sum(dim=1)).sum() \
               + (cost_s - 2).sum()

        # cost_s.detach() in 39
        # compute gradient's variance loss w.r.t. surrogate's parameter
        grad_p = torch.autograd.grad(loss, policy_net.parameters(),
                                     grad_outputs=torch.ones_like(loss), create_graph=True, retain_graph=True)
        grad_temp = torch.cat([torch.reshape(p, [-1]) for p in grad_p], 0)
        grad_ps = torch.square(grad_temp).mean(0)
        grad_s = torch.autograd.grad(grad_ps, surrogate.parameters(),
                                     grad_outputs=torch.ones_like(grad_ps), retain_graph=True, allow_unused=True)
        # Optimize the policy net
        optim_p.zero_grad()
        loss.backward()
        optim_p.step()
        # Optimize the surrogate net
        optim_s.zero_grad()
        for params, grad in zip(surrogate.parameters(), grad_s):
            params.grad = grad
        optim_s.step()
        if itr % 100 == 0:
            print('\nIteration:', itr)
        print(format(sum(cost) / batch_size, '.4f'))

        # validate and save best nets
        if (itr+1) % 100 == 0:
            validation_result = validate(validation_data, policy_net, no_agent, device)
            if validation_result < best_so_far:
                torch.save(policy_net.state_dict(), './saved_model/{}_{}.pth'.format(str(no_nodes), str(no_agent)))
                print('Found better policy, and the validation result is:', format(validation_result, '.4f'))
                validation_results.append(validation_result)
                best_so_far = validation_result
    return validation_results


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(2)

    n_agent = 5
    n_nodes = 50
    batch_size = 512
    lr_p = 1e-4
    lr_s = 1e-4
    iteration = 3000

    policy = Policy(in_chnl=2, hid_chnl=32, n_agent=n_agent, key_size_embd=64,
                    key_size_policy=64, val_size=64, clipping=10, dev=dev)
    
    surrogate = Surrogate(in_dim=n_nodes-1, out_dim=1, n_hidden=128, nonlin='relu', dev=dev)

    best_results = train(batch_size, n_nodes, policy, surrogate, lr_p, lr_s, n_agent, iteration, dev)
    print(min(best_results))
