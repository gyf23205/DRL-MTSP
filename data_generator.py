import torch


def data_gen(no_nodes, batch_size, flag):
    if flag == 'validation':
        torch.save(torch.rand(size=[batch_size, no_nodes, 2]), './validation_data/validation_data_'+str(no_nodes)+'_'+str(batch_size))
    elif flag == 'testing':
        torch.save(torch.rand(size=[batch_size, no_nodes, 2]), './testing_data/testing_data_'+str(no_nodes)+'_'+str(batch_size))
    elif flag == 'training':
        torch.save(torch.rand(size=[batch_size, no_nodes, 2]),
                   './training_data/training_data_' + str(no_nodes) + '_' + str(batch_size))
    else:
        print('flag should be "training", "testing", or "validation".')


if __name__ == '__main__':
    n_nodes = 50
    b_size = 512
    flag = 'training'
    torch.manual_seed(4)

    data_gen(n_nodes, b_size, flag)