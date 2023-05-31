import torch
from torch.distributions import Categorical
from ortools_tsp import my_solve


def plan2tensor(plan, batch, num_city):
    plan_tensor = torch.ones([batch, num_city]) * -1
    plan_tensor[:, 0] = 0
    for i in range(batch):
        for j, c in enumerate(plan[i]):
            if c != 0:
                plan_tensor[i, c] = j
    return plan_tensor


def tensor2plan(plan_tensor, batch):
    maximum = torch.max(plan_tensor, dim=1)
    plan = [[0 for _ in range(int(maximum.values[i] + 2))] for i in range(batch)]
    for i in range(batch):
        for j, order in enumerate(plan_tensor[i, :]):
            if order != -1:
                plan[i][int(order)] = j
    return plan


class DiffCO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pi, data):
        # Get actions
        dist = Categorical(pi.transpose(2, 1))
        action_int = dist.sample()

        # Get allocations
        allocations = [[[] for _ in range(pi.shape[1])] for _ in range(pi.shape[0])]
        for i in range(pi.shape[0]):
            for a in allocations[i]:
                a.append(0)
            for j, n in enumerate(action_int.tolist()[i]):
                allocations[i][n].append(j+1)


        # Get solutions
        subtour_max_cost = [0 for _ in range(data.shape[0])]
        obj_list = [0 for _ in range(data.shape[0])]
        subtour_max_city_index = [None for _ in range(data.shape[0])]
        subtour_max_agent_index = [-1 for _ in range(data.shape[0])]
        data = data * 1000
        depot = data[:, 0, :].tolist()
        sub_tours = [[[] for _ in range(pi.shape[1])] for _ in range(pi.shape[0])]  # Get instances
        for i in range(data.shape[0]):
            for tour in sub_tours[i]:
                tour.append(depot[i])
            for n, m in zip(action_int.tolist()[i], data.tolist()[i][1:]):
                sub_tours[i][n].append(m)

        for i in range(data.shape[0]):
            for a in range(pi.shape[1]):
                instance = sub_tours[i][a]
                city_indices = allocations[i][a]
                sub_tour_cost, sub_tour_route, obj_value = my_solve(instance, city_indices)
                if sub_tour_cost >= subtour_max_cost[i]:
                    subtour_max_cost[i] = sub_tour_cost
                    obj_list[i] = obj_value
                    subtour_max_city_index[i] = sub_tour_route
                    subtour_max_agent_index[i] = a
        subtour_max_city_index_tensor = plan2tensor(subtour_max_city_index, batch=data.shape[0], num_city=data.shape[1])
        ctx.save_for_backward(pi, data, subtour_max_city_index_tensor, torch.tensor(subtour_max_agent_index))
        subtour_max_cost = torch.tensor(subtour_max_cost, dtype=torch.float32)
        obj_list = torch.tensor(obj_list, dtype=torch.float32)
        return subtour_max_cost, action_int, obj_list

    @staticmethod
    def backward(ctx, dloss_dcost, daction, dobj):
        print(111)
        batch = dloss_dcost.shape[0]
        pi, data, sub_tour_route, agent_num = ctx.saved_tensors
        sub_tour_route = tensor2plan(sub_tour_route, batch)
        diff = [[] for _ in range(batch)]
        for i in range(batch):
            for j in range(len(sub_tour_route[i])-1):
                diff[i].append(2 * (sub_tour_route[i][j+1] - sub_tour_route[i][j]))
        dcost_dindices = [[] for _ in range(batch)]
        for i in range(batch):
            for j in range(1, len(diff[i])):
                dcost_dindices[i].append((diff[i][j-1] - diff[i][j]) * dloss_dcost[i])
        grad = torch.zeros_like(pi)
        grad_data = torch.zeros_like(data)
        for i in range(len(dcost_dindices)):
            for j in range(1, len(sub_tour_route[i])-1):
                a = agent_num[i]
                c = sub_tour_route[i][j]
                if dcost_dindices[i][j-1] < 0:
                    grad[i, a, :c-1] += dcost_dindices[i][j-1]
                    grad[i, a, c-1] -= dcost_dindices[i][j-1]
                elif dcost_dindices[i][j-1] > 0:
                    grad[i, a, c:] -= dcost_dindices[i][j-1]
                    grad[i, a, c-1] += dcost_dindices[i][j-1]
                else:
                    pass
        return grad, grad_data


class PI(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pi = torch.nn.Parameter(torch.rand([10, 5, 14]))


if __name__ == '__main__':
    # pi = torch.nn.Linear(10, 5)
    # pi = torch.rand([10, 5, 14], requires_grad=True).to('cuda')
    pi = PI()
    # pi_copy = torch.clone(pi)
    # pi.requires_grad = True
    optimizer = torch.optim.SGD(pi.parameters(), lr=0.01)
    data = torch.rand([10, 15, 2], requires_grad=True).to('cuda')
    diff_allo = DiffCO.apply
    subtour_max_cost, action_int, obj = diff_allo(pi.pi, data)
    # subtour_max_cost.requires_grad = True
    loss = torch.sum(subtour_max_cost.clone().to('cuda'))
    loss.backward()
    print(pi.pi.grad)
    optimizer.step()
    # print(pi - pi_copy)


