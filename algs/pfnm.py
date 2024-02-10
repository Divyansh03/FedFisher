### Code adopted from https://github.com/IBM/FedMA

from lapsolver import solve_dense
import copy
import torch
import time
import logging
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class SimpleCNNContainerConvBlocks(nn.Module):
    def __init__(self, input_channel, num_filters, kernel_size, output_dim=10):
        super(SimpleCNNContainerConvBlocks, self).__init__()
        '''
        A testing cnn container, which allows initializing a CNN with given dims
        We use this one to estimate matched output of conv blocks
        num_filters (list) :: number of convolution filters
        hidden_dims (list) :: number of neurons in hidden layers
        Assumptions:
        i) we use only two conv layers and three hidden layers (including the output layer)
        ii) kernel size in the two conv layers are identical
        '''
        self.conv1 = nn.Conv2d(input_channel, num_filters[0], kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x


class SimpleCNNCIFARContainerConvBlocks(nn.Module):
    def __init__(self, input_channel, num_filters, kernel_size, output_dim=10):
        super(SimpleCNNCIFARContainerConvBlocks, self).__init__()
        '''
        A testing cnn container, which allows initializing a CNN with given dims
        We use this one to estimate matched output of conv blocks
        num_filters (list) :: number of convolution filters
        hidden_dims (list) :: number of neurons in hidden layers
        Assumptions:
        i) we use only two conv layers and three hidden layers (including the output layer)
        ii) kernel size in the two conv layers are identical
        '''
        self.conv1 = nn.Conv2d(input_channel, num_filters[0], kernel_size, padding = 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size, padding = 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

class LeNetContainer(nn.Module):
    def __init__(self, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(LeNetContainer, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size, 1)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size, 1)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], output_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, x.size()[1]*x.size()[2]*x.size()[3])
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ModerateCNNContainerConvBlocksMNIST(nn.Module):
    def __init__(self, num_filters, output_dim=10):
        super(ModerateCNNContainerConvBlocksMNIST, self).__init__()

        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=num_filters[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[2], out_channels=num_filters[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=num_filters[3], out_channels=num_filters[4], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[4], out_channels=num_filters[5], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        return x




class ModerateCNN(nn.Module):
    def __init__(self, output_dim = 10):
        super(ModerateCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            #nn.Linear(4096, 1024),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            #nn.Linear(1024, 512),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x



class ModerateCNNContainerConvBlocks(nn.Module):
    def __init__(self, num_filters, output_dim=10):
        super(ModerateCNNContainerConvBlocks, self).__init__()

        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=num_filters[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[2], out_channels=num_filters[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=num_filters[3], out_channels=num_filters[4], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[4], out_channels=num_filters[5], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        return x






def row_param_cost(global_weights, weights_j_l, global_sigmas, sigma_inv_j):

    match_norms = ((weights_j_l + global_weights) ** 2 / (sigma_inv_j + global_sigmas)).sum(axis=1) - (
                global_weights ** 2 / global_sigmas).sum(axis=1)

    return match_norms


def row_param_cost_simplified(global_weights, weights_j_l, sij_p_gs, red_term):
    match_norms = ((weights_j_l + global_weights) ** 2 / sij_p_gs).sum(axis=1) - red_term
    return match_norms




def rpc_parallel(global_weights, weights_j, sij_p_gs, red_term, iter_range):
    res = np.array([(((weights_j[ir] + global_weights) ** 2 / sij_p_gs).sum(axis=1) - red_term) for ir in iter_range])
    return res


def compute_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                 popularity_counts, gamma, J):

    param_cost_start = time.time()
    Lj = weights_j.shape[0]
    counts = np.minimum(np.array(popularity_counts, dtype=np.float32), 10)

    sij_p_gs = sigma_inv_j + global_sigmas
    red_term = (global_weights ** 2 / global_sigmas).sum(axis=1)
    stupid_line_start = time.time()

    param_cost = np.array([row_param_cost_simplified(global_weights, weights_j[l], sij_p_gs, red_term) for l in range(Lj)], dtype=np.float32)
    
    stupid_line_dur = time.time() - stupid_line_start

    param_cost += np.log(counts / (J - counts))
    param_cost_dur = time.time() - param_cost_start

    ## Nonparametric cost
    nonparam_start = time.time()
    L = global_weights.shape[0]
    max_added = min(Lj, max(700 - L, 1))
    nonparam_cost = np.outer((((weights_j + prior_mean_norm) ** 2 / (prior_inv_sigma + sigma_inv_j)).sum(axis=1) - (
                prior_mean_norm ** 2 / prior_inv_sigma).sum()), np.ones(max_added, dtype=np.float32))
    cost_pois = 2 * np.log(np.arange(1, max_added + 1))
    nonparam_cost -= cost_pois
    nonparam_cost += 2 * np.log(gamma / J)

    nonparam_dur = time.time() - nonparam_start
   
    full_cost = np.hstack((param_cost, nonparam_cost)).astype(np.float32)
    return full_cost


def matching_upd_j(weights_j, global_weights, sigma_inv_j, global_sigmas, prior_mean_norm, prior_inv_sigma,
                   popularity_counts, gamma, J):

    L = global_weights.shape[0]

    compute_cost_start = time.time()
    full_cost = compute_cost(global_weights.astype(np.float32), weights_j.astype(np.float32), global_sigmas.astype(np.float32), sigma_inv_j.astype(np.float32), prior_mean_norm.astype(np.float32), prior_inv_sigma.astype(np.float32),
                             popularity_counts, gamma, J)
    compute_cost_dur = time.time() - compute_cost_start
    
    start_time = time.time()
    row_ind, col_ind = solve_dense(-full_cost)
    solve_dur = time.time() - start_time


    assignment_j = []

    new_L = L

    for l, i in zip(row_ind, col_ind):
        if i < L:
            popularity_counts[i] += 1
            assignment_j.append(i)
            global_weights[i] += weights_j[l]
            global_sigmas[i] += sigma_inv_j
        else:  # new neuron
            popularity_counts += [1]
            assignment_j.append(new_L)
            new_L += 1
            global_weights = np.vstack((global_weights, prior_mean_norm + weights_j[l]))
            global_sigmas = np.vstack((global_sigmas, prior_inv_sigma + sigma_inv_j))

    return global_weights, global_sigmas, popularity_counts, assignment_j


def objective(global_weights, global_sigmas):
    obj = ((global_weights) ** 2 / global_sigmas).sum()
    return obj


def patch_weights(w_j, L_next, assignment_j_c):
    if assignment_j_c is None:
        return w_j
    new_w_j = np.zeros((w_j.shape[0], L_next))
    new_w_j[:, assignment_j_c] = w_j
    return new_w_j

def block_patching(w_j, L_next, assignment_j_c, layer_index, model_meta_data, 
                                matching_shapes=None, 
                                layer_type="fc", 
                                dataset="cifar10",
                                network_name="lenet"):
    """
    In CNN, weights patching needs to be handled block-wisely
    We handle all conv layers and the first fc layer connected with the output of conv layers here
    """


    if assignment_j_c is None:
        return w_j

    layer_meta_data = model_meta_data[2 * layer_index - 2]
    prev_layer_meta_data = model_meta_data[2 * layer_index - 2 - 2]

    if layer_type == "conv":    
        new_w_j = np.zeros((w_j.shape[0], L_next*(layer_meta_data[-1]**2)))

        # we generate a sequence of block indices
        block_indices = [np.arange(i*layer_meta_data[-1]**2, (i+1)*layer_meta_data[-1]**2) for i in range(L_next)]
        ori_block_indices = [np.arange(i*layer_meta_data[-1]**2, (i+1)*layer_meta_data[-1]**2) for i in range(layer_meta_data[1])]
        for ori_id in range(layer_meta_data[1]):
            new_w_j[:, block_indices[assignment_j_c[ori_id]]] = w_j[:, ori_block_indices[ori_id]]

    elif layer_type == "fc":
        # we need to estimate the output shape here:
        if network_name == "simple-cnn":
            if dataset in ("cifar10", "cinic10", "svhn", "cifar100"):
                shape_estimator = SimpleCNNContainerConvBlocks(input_channel=3, num_filters=matching_shapes, kernel_size=5)
            elif dataset == "mnist":
                shape_estimator = SimpleCNNContainerConvBlocks(input_channel=1, num_filters=matching_shapes, kernel_size=5)
        if network_name == "simple-cnn-cifar":
            if dataset in ("cifar10", "cinic10", "svhn", "cifar100"):
                shape_estimator = SimpleCNNCIFARContainerConvBlocks(input_channel=3, num_filters=matching_shapes, kernel_size=5)
            elif dataset == "mnist":
                shape_estimator = SimpleCNNCIFARContainerConvBlocks(input_channel=1, num_filters=matching_shapes, kernel_size=5)
        elif network_name == "moderate-cnn":
            if dataset in ("cifar10", "cinic10"):
                shape_estimator = ModerateCNNContainerConvBlocks(num_filters=matching_shapes)
            elif dataset == "mnist":
                shape_estimator = ModerateCNNContainerConvBlocksMNIST(num_filters=matching_shapes)
        elif network_name == "lenet":
            shape_estimator = LeNetContainer(num_filters=matching_shapes, kernel_size=5)

        if dataset in ("cifar10", "cinic10" , "svhn", "cifar100"):
            dummy_input = torch.rand(1, 3, 32, 32)
        elif dataset == "mnist":
            dummy_input = torch.rand(1, 1, 28, 28)
        estimated_output = shape_estimator(dummy_input)
        new_w_j = np.zeros((w_j.shape[0], estimated_output.view(-1).size()[0]))
        logger.info("estimated_output shape : {}".format(estimated_output.size()))
        logger.info("meta data of previous layer: {}".format(prev_layer_meta_data))
        
        block_indices = [np.arange(i*estimated_output.size()[-1]**2, (i+1)*estimated_output.size()[-1]**2) for i in range(L_next)]
       
        ori_block_indices = [np.arange(i*estimated_output.size()[-1]**2, (i+1)*estimated_output.size()[-1]**2) for i in range(prev_layer_meta_data[0])]
        
        for ori_id in range(prev_layer_meta_data[0]):

            new_w_j[:, block_indices[assignment_j_c[ori_id]]] = w_j[:, ori_block_indices[ori_id]]
    return new_w_j


def process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0):
    J = len(batch_weights)
    sigma_bias = sigma
    sigma0_bias = sigma0
    mu0_bias = 0.1
    softmax_bias = [batch_weights[j][-1] for j in range(J)]
    softmax_inv_sigma = [s / sigma_bias for s in last_layer_const]
    softmax_bias = sum([b * s for b, s in zip(softmax_bias, softmax_inv_sigma)]) + mu0_bias / sigma0_bias
    softmax_inv_sigma = 1 / sigma0_bias + sum(softmax_inv_sigma)
    return softmax_bias, softmax_inv_sigma


def match_layer(weights_bias, sigma_inv_layer, mean_prior, sigma_inv_prior, gamma, it):
    J = len(weights_bias)

    group_order = sorted(range(J), key=lambda x: -weights_bias[x].shape[0])

    batch_weights_norm = [w * s for w, s in zip(weights_bias, sigma_inv_layer)]
    prior_mean_norm = mean_prior * sigma_inv_prior

    global_weights = prior_mean_norm + batch_weights_norm[group_order[0]]
    global_sigmas = np.outer(np.ones(global_weights.shape[0]), sigma_inv_prior + sigma_inv_layer[group_order[0]])

    popularity_counts = [1] * global_weights.shape[0]

    assignment = [[] for _ in range(J)]

    assignment[group_order[0]] = list(range(global_weights.shape[0]))

    ## Initialize
    for j in group_order[1:]:
        global_weights, global_sigmas, popularity_counts, assignment_j = matching_upd_j(batch_weights_norm[j],
                                                                                        global_weights,
                                                                                        sigma_inv_layer[j],
                                                                                        global_sigmas, prior_mean_norm,
                                                                                        sigma_inv_prior,
                                                                                        popularity_counts, gamma, J)
        assignment[j] = assignment_j

    ## Iterate over groups
    for iteration in range(it):
        random_order = np.random.permutation(J)
        for j in random_order:  # random_order:
            to_delete = []
            Lj = len(assignment[j])
            for l, i in sorted(zip(range(Lj), assignment[j]), key=lambda x: -x[1]):
                popularity_counts[i] -= 1
                if popularity_counts[i] == 0:
                    del popularity_counts[i]
                    to_delete.append(i)
                    for j_clean in range(J):
                        for idx, l_ind in enumerate(assignment[j_clean]):
                            if i < l_ind and j_clean != j:
                                assignment[j_clean][idx] -= 1
                            elif i == l_ind and j_clean != j:
                                logger.info('Warning - weird unmatching')
                else:
                    global_weights[i] = global_weights[i] - batch_weights_norm[j][l]
                    global_sigmas[i] -= sigma_inv_layer[j]

            global_weights = np.delete(global_weights, to_delete, axis=0)
            global_sigmas = np.delete(global_sigmas, to_delete, axis=0)

            ## Match j
            global_weights, global_sigmas, popularity_counts, assignment_j = matching_upd_j(batch_weights_norm[j],
                                                                                            global_weights,
                                                                                            sigma_inv_layer[j],
                                                                                            global_sigmas,
                                                                                            prior_mean_norm,
                                                                                            sigma_inv_prior,
                                                                                            popularity_counts, gamma, J)
            assignment[j] = assignment_j

    logger.info('Number of global neurons is %d, gamma %f' % (global_weights.shape[0], gamma))
    logger.info("***************Shape of global weights after match: {} ******************".format(global_weights.shape))
    return assignment, global_weights, global_sigmas
    



def layer_wise_group_descent(batch_weights, layer_index, batch_frequencies, sigma_layers, 
                                sigma0_layers, gamma_layers, it, 
                                model_meta_data, 
                                model_layer_type,
                                n_layers,
                                matching_shapes,
                                args):
    """
    We implement a layer-wise matching here:
    """
    if type(sigma_layers) is not list:
        sigma_layers = (n_layers - 1) * [sigma_layers]
    if type(sigma0_layers) is not list:
        sigma0_layers = (n_layers - 1) * [sigma0_layers]
    if type(gamma_layers) is not list:
        gamma_layers = (n_layers - 1) * [gamma_layers]

    last_layer_const = []
    total_freq = sum(batch_frequencies)
    for f in batch_frequencies:
        last_layer_const.append(f / total_freq)

    # J: number of workers
    J = len(batch_weights)
    # init_num_kernel: the number of conv filters in the first conv layer 
    init_num_kernel = batch_weights[0][0].shape[0]

    # for saving (#channel * k * k)
    init_channel_kernel_dims = []
    for bw in batch_weights[0]:
        if len(bw.shape) > 1:
            init_channel_kernel_dims.append(bw.shape[1])
    logger.info("init_channel_kernel_dims: {}".format(init_channel_kernel_dims))
    
    sigma_bias_layers = sigma_layers
    sigma0_bias_layers = sigma0_layers
    mu0 = 0.
    mu0_bias = 0.1
    assignment_c = [None for j in range(J)]
    L_next = None

    sigma = sigma_layers[layer_index - 1]
    sigma_bias = sigma_bias_layers[layer_index - 1]
    gamma = gamma_layers[layer_index - 1]
    sigma0 = sigma0_layers[layer_index - 1]
    sigma0_bias = sigma0_bias_layers[layer_index - 1]

    if layer_index <= 1:
        weights_bias = [np.hstack((batch_weights[j][0], batch_weights[j][layer_index * 2 - 1].reshape(-1, 1))) for j in range(J)]

        sigma_inv_prior = np.array(
            init_channel_kernel_dims[layer_index - 1] * [1 / sigma0] + [1 / sigma0_bias])
        mean_prior = np.array(init_channel_kernel_dims[layer_index - 1] * [mu0] + [mu0_bias])

        # handling 2-layer neural network
        if n_layers == 2:
            sigma_inv_layer = [
                np.array(D * [1 / sigma] + [1 / sigma_bias] + [y / sigma for y in last_layer_const[j]]) for j in range(J)]
        else:
            sigma_inv_layer = [np.array(init_channel_kernel_dims[layer_index - 1] * [1 / sigma] + [1 / sigma_bias]) for j in range(J)]

    elif layer_index == (n_layers - 1) and n_layers > 2:
        # our assumption is that this branch will consistently handle the last fc layers
        layer_type = model_layer_type[2 * layer_index - 2]
        prev_layer_type = model_layer_type[2 * layer_index - 2 - 2]
        first_fc_identifier = (('fc' in layer_type or 'classifier' in layer_type) and ('conv' in prev_layer_type or 'features' in layer_type))

        # we switch to ignore the last layer here:
        if first_fc_identifier:
            weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2].T, 
                                        batch_weights[j][2 * layer_index - 1].reshape(-1, 1))) for j in range(J)]
        else:
            weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2].T, 
                                        batch_weights[j][2 * layer_index - 1].reshape(-1, 1))) for j in range(J)]


        sigma_inv_prior = np.array([1 / sigma0_bias] + (weights_bias[0].shape[1] - 1) * [1 / sigma0])
        mean_prior = np.array([mu0_bias] + (weights_bias[0].shape[1] - 1) * [mu0])


        sigma_inv_layer = [np.array([1 / sigma_bias] + (weights_bias[j].shape[1] - 1) * [1 / sigma]) for j in range(J)]

    elif (layer_index > 1 and layer_index < (n_layers - 1)):
        layer_type = model_layer_type[2 * layer_index - 2]
        prev_layer_type = model_layer_type[2 * layer_index - 2 - 2]

        if 'conv' in layer_type or 'features' in layer_type:
            weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2], batch_weights[j][2 * layer_index - 1].reshape(-1, 1))) for j in range(J)]

        elif 'fc' in layer_type or 'classifier' in layer_type:
            # we need to determine if the type of the current layer is the same as it's previous layer
            # i.e. we need to identify if the fully connected layer we're working on is the first fc layer after the conv block
            #first_fc_identifier = ('fc' in layer_type and 'conv' in prev_layer_type)
            first_fc_identifier = (('fc' in layer_type or 'classifier' in layer_type) and ('conv' in prev_layer_type or 'features' in layer_type))
            #logger.info("first_fc_identifier: {}".format(first_fc_identifier))
            if first_fc_identifier:
                weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2].T, batch_weights[j][2 * layer_index - 1].reshape(-1, 1))) for j in range(J)]
            else:
                weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2].T, batch_weights[j][2 * layer_index - 1].reshape(-1, 1))) for j in range(J)]          

        sigma_inv_prior = np.array([1 / sigma0_bias] + (weights_bias[0].shape[1] - 1) * [1 / sigma0])
        mean_prior = np.array([mu0_bias] + (weights_bias[0].shape[1] - 1) * [mu0])
        sigma_inv_layer = [np.array([1 / sigma_bias] + (weights_bias[j].shape[1] - 1) * [1 / sigma]) for j in range(J)]

    logger.info("Layer index: {}, init_num_kernel: {}".format(layer_index, init_num_kernel))
    logger.info("weights bias: {}".format(weights_bias[0].shape))
    logger.info("sigma_inv_prior shape: {}".format(sigma_inv_prior.shape))
    logger.info("mean_prior shape: {}".format(mean_prior.shape))
    logger.info("sigma_inv_layer shape: {}".format(sigma_inv_layer[0].shape))

    assignment_c, global_weights_c, global_sigmas_c = match_layer(weights_bias, sigma_inv_layer, mean_prior,
                                                                  sigma_inv_prior, gamma, it)

    L_next = global_weights_c.shape[0]

    if layer_index <= 1:
        if n_layers == 2:
            softmax_bias, softmax_inv_sigma = process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0)
            global_weights_out = [softmax_bias]
            global_inv_sigmas_out = [softmax_inv_sigma]
        
        global_weights_out = [global_weights_c[:, :init_channel_kernel_dims[int(layer_index/2)]], global_weights_c[:, init_channel_kernel_dims[int(layer_index/2)]]]
        global_inv_sigmas_out = [global_sigmas_c[:, :init_channel_kernel_dims[int(layer_index/2)]], global_sigmas_c[:, init_channel_kernel_dims[int(layer_index/2)]]]

        logger.info("Branch A, Layer index: {}, Global weights out shapes: {}".format(layer_index, [gwo.shape for gwo in global_weights_out]))

    elif layer_index == (n_layers - 1) and n_layers > 2:
        softmax_bias, softmax_inv_sigma = process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0)

        layer_type = model_layer_type[2 * layer_index - 2]
        prev_layer_type = model_layer_type[2 * layer_index - 2 - 2]
        #first_fc_identifier = ('fc' in layer_type and 'conv' in prev_layer_type)
        first_fc_identifier = (('fc' in layer_type or 'classifier' in layer_type) and ('conv' in prev_layer_type or 'features' in layer_type))


        layer_type = model_layer_type[2 * layer_index - 2]
        gwc_shape = global_weights_c.shape
        if "conv" in layer_type or 'features' in layer_type:
            global_weights_out = [global_weights_c[:, 0:gwc_shape[1]-1], global_weights_c[:, gwc_shape[1]-1]]
            global_inv_sigmas_out = [global_sigmas_c[:, 0:gwc_shape[1]-1], global_sigmas_c[:, gwc_shape[1]-1]]
        elif "fc" in layer_type or 'classifier' in layer_type:
            global_weights_out = [global_weights_c[:, 0:gwc_shape[1]-1].T, global_weights_c[:, gwc_shape[1]-1]]
            global_inv_sigmas_out = [global_sigmas_c[:, 0:gwc_shape[1]-1].T, global_sigmas_c[:, gwc_shape[1]-1]]

        logger.info("#### Branch B, Layer index: {}, Global weights out shapes: {}".format(layer_index, [gwo.shape for gwo in global_weights_out]))

    elif (layer_index > 1 and layer_index < (n_layers - 1)):
        layer_type = model_layer_type[2 * layer_index - 2]
        gwc_shape = global_weights_c.shape

        if "conv" in layer_type or 'features' in layer_type:
            global_weights_out = [global_weights_c[:, 0:gwc_shape[1]-1], global_weights_c[:, gwc_shape[1]-1]]
            global_inv_sigmas_out = [global_sigmas_c[:, 0:gwc_shape[1]-1], global_sigmas_c[:, gwc_shape[1]-1]]
        elif "fc" in layer_type or 'classifier' in layer_type:
            global_weights_out = [global_weights_c[:, 0:gwc_shape[1]-1].T, global_weights_c[:, gwc_shape[1]-1]]
            global_inv_sigmas_out = [global_sigmas_c[:, 0:gwc_shape[1]-1].T, global_sigmas_c[:, gwc_shape[1]-1]]
        logger.info("Branch layer index, Layer index: {}, Global weights out shapes: {}".format(layer_index, [gwo.shape for gwo in global_weights_out]))

    logger.info("global inv sigma out shape: {}".format([giso.shape for giso in global_inv_sigmas_out]))
    map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]
    return map_out, assignment_c, L_next



#@title PFNM CNN code


class SimpleCNNContainer(nn.Module):
    def __init__(self, input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNContainer, self).__init__()
        '''
        A testing cnn container, which allows initializing a CNN with given dims
        num_filters (list) :: number of convolution filters
        hidden_dims (list) :: number of neurons in hidden layers
        Assumptions:
        i) we use only two conv layers and three hidden layers (including the output layer)
        ii) kernel size in the two conv layers are identical
        '''
        self.conv1 = nn.Conv2d(input_channel, num_filters[0], kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size)
        
        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.size()[1]*x.size()[2]*x.size()[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleCNNCIFARContainer(nn.Module):
    def __init__(self, input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNCIFARContainer, self).__init__()
        '''
        A testing cnn container, which allows initializing a CNN with given dims
        num_filters (list) :: number of convolution filters
        hidden_dims (list) :: number of neurons in hidden layers
        Assumptions:
        i) we use only two conv layers and three hidden layers (including the output layer)
        ii) kernel size in the two conv layers are identical
        '''
        self.conv1 = nn.Conv2d(input_channel, num_filters[0],  kernel_size, padding = 2,)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1],  kernel_size, padding = 2,)
        
        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.size()[1]*x.size()[2]*x.size()[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ModerateCNNContainer(nn.Module):
    def __init__(self, input_channels, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(ModerateCNNContainer, self).__init__()

        ##
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=input_channels, out_channels=num_filters[0], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[2], out_channels=num_filters[3], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=num_filters[3], out_channels=num_filters[4], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[4], out_channels=num_filters[5], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dims[1], output_dim)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

    def forward_conv(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        return x

def trans_next_conv_layer_forward(layer_weight, next_layer_shape):
    reshaped = layer_weight.reshape(next_layer_shape).transpose((1, 0, 2, 3)).reshape((next_layer_shape[1], -1))
    return reshaped

def trans_next_conv_layer_backward(layer_weight, next_layer_shape):
    reconstructed_next_layer_shape = (next_layer_shape[1], next_layer_shape[0], next_layer_shape[2], next_layer_shape[3])
    reshaped = layer_weight.reshape(reconstructed_next_layer_shape).transpose(1, 0, 2, 3).reshape(next_layer_shape[0], -1)
    return reshaped




def local_retrain_dummy(weights, args, mode="bottom-up", freezing_index=0, ori_assignments=None, device="cpu"):
    """
    FOR FPNM
    """
    if args.model == "lenet":
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        kernel_size = 5
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1]]
        output_dim = weights[-1].shape[0]
        logger.info("Num filters: {}, Input dim: {}, hidden_dims: {}, output_dim: {}".format(num_filters, input_dim, hidden_dims, output_dim))

        matched_cnn = LeNetContainer(
                                    num_filters=num_filters,
                                    kernel_size=kernel_size,
                                    input_dim=input_dim,
                                    hidden_dims=hidden_dims,
                                    output_dim=output_dim)
    elif args.model == "vgg":
        matched_shapes = [w.shape for w in weights]
        matched_cnn = matched_vgg11(matched_shapes=matched_shapes)
    elif args.model == "simple-cnn":
        # input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        # [(9, 75), (9,), (19, 225), (19,), (475, 123), (123,), (123, 87), (87,), (87, 10), (10,)]
        if args.dataset in ("cifar10", "cinic10"):
            input_channel = 3
        elif args.dataset == "mnist":
            input_channel = 1

        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1], weights[6].shape[1]]
        matched_cnn = SimpleCNNContainer(input_channel=input_channel, 
                                        num_filters=num_filters, 
                                        kernel_size=5, 
                                        input_dim=input_dim, 
                                        hidden_dims=hidden_dims, 
                                        output_dim=10)

    elif args.model == "simple-cnn-cifar":
        # input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        # [(9, 75), (9,), (19, 225), (19,), (475, 123), (123,), (123, 87), (87,), (87, 10), (10,)]
        if args.dataset in ("cifar10", "cinic10"):
            input_channel = 3
        elif args.dataset == "mnist":
            input_channel = 1

        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1], weights[6].shape[1]]
        matched_cnn = SimpleCNNCIFARContainer(input_channel=input_channel, 
                                        num_filters=num_filters, 
                                        kernel_size=5, 
                                        input_dim=input_dim, 
                                        hidden_dims=hidden_dims, 
                                        output_dim=10)

    elif args.model == "moderate-cnn":
        #[(35, 27), (35,), (68, 315), (68,), (132, 612), (132,), (132, 1188), (132,), 
        #(260, 1188), (260,), (260, 2340), (260,), 
        #(4160, 1025), (1025,), (1025, 515), (515,), (515, 10), (10,)]
        if mode not in ("block-wise", "squeezing"):
            num_filters = [weights[0].shape[0], weights[2].shape[0], weights[4].shape[0], weights[6].shape[0], weights[8].shape[0], weights[10].shape[0]]
            input_dim = weights[12].shape[0]
            hidden_dims = [weights[12].shape[1], weights[14].shape[1]]

            input_dim = weights[12].shape[0]
        elif mode == "block-wise":
            # for block-wise retraining the `freezing_index` becomes a range of indices
            # so at here we need to generate a unfreezing list:
            __unfreezing_list = []
            for fi in freezing_index:
                __unfreezing_list.append(2*fi-2)
                __unfreezing_list.append(2*fi-1)

            # we need to do two changes here:
            # i) switch the number of filters in the freezing indices block to the original size
            # ii) cut the correspoidng color channels
            __fixed_indices = set([i*2 for i in range(6)]) # 0, 2, 4, 6, 8, 10
            dummy_model = ModerateCNN()

            num_filters = []
            for pi, param in enumerate(dummy_model.parameters()):
                if pi in __fixed_indices:
                    if pi in __unfreezing_list:
                        num_filters.append(param.size()[0])
                    else:
                        num_filters.append(weights[pi].shape[0])
            del dummy_model
            logger.info("################ Num filters for now are : {}".format(num_filters))
            # note that we hard coded index of the last conv layer here to make sure the dimension is compatible
            if freezing_index[0] != 6:
            #if freezing_index[0] not in (6, 7):
                input_dim = weights[12].shape[0]
            else:
                # we need to estimate the output shape here:
                shape_estimator = ModerateCNNContainerConvBlocks(num_filters=num_filters)
                dummy_input = torch.rand(1, 3, 32, 32)
                estimated_output = shape_estimator(dummy_input)
                #estimated_shape = (estimated_output[1], estimated_output[2], estimated_output[3])
                input_dim = estimated_output.view(-1).size()[0]

            if (freezing_index[0] <= 6) or (freezing_index[0] > 8):
                hidden_dims = [weights[12].shape[1], weights[14].shape[1]]
            else:
                dummy_model = ModerateCNN(output_dim = args.n_c)
                for pi, param in enumerate(dummy_model.parameters()):
                    if pi == 2*freezing_index[0] - 2:
                        _desired_shape = param.size()[0]
                if freezing_index[0] == 7:
                    hidden_dims = [_desired_shape, weights[14].shape[1]]
                elif freezing_index[0] == 8:
                    hidden_dims = [weights[12].shape[1], _desired_shape]
        elif mode == "squeezing":
            pass


        if args.dataset in ("cifar10", "cinic10"):
            if mode == "squeezing":
                matched_cnn = ModerateCNN(output_dim = args.n_c)
            else:
                matched_cnn = ModerateCNNContainer(3,
                                                    num_filters, 
                                                    kernel_size=3, 
                                                    input_dim=input_dim, 
                                                    hidden_dims=hidden_dims, 
                                                    output_dim=args.n_c)
        elif args.dataset == "mnist":
            matched_cnn = ModerateCNNContainer(1,
                                                num_filters, 
                                                kernel_size=3, 
                                                input_dim=input_dim, 
                                                hidden_dims=hidden_dims, 
                                                output_dim=args.n_c)
    
    new_state_dict = {}
    model_counter = 0
    n_layers = int(len(weights) / 2)

    # we hardcoded this for now: will probably make changes later
    #if mode != "block-wise":
    if mode not in ("block-wise", "squeezing"):
        __non_loading_indices = []
    else:
        if mode == "block-wise":
            if freezing_index[0] != n_layers:
                __non_loading_indices = copy.deepcopy(__unfreezing_list)
                __non_loading_indices.append(__unfreezing_list[-1]+1) # add the index of the weight connects to the next layer
            else:
                __non_loading_indices = copy.deepcopy(__unfreezing_list)
        elif mode == "squeezing":
            # please note that at here we need to reconstruct the entire local network and retrain it
            __non_loading_indices = [i for i in range(len(weights))]

    def __reconstruct_weights(weight, assignment, layer_ori_shape, matched_num_filters=None, weight_type="conv_weight", slice_dim="filter"):
        # what contains in the param `assignment` is the assignment for a certain layer, a certain worker
        """
        para:: slice_dim: for reconstructing the conv layers, for each of the three consecutive layers, we need to slice the 
               filter/kernel to reconstruct the first conv layer; for the third layer in the consecutive block, we need to 
               slice the color channel 
        """
        if weight_type == "conv_weight":
            if slice_dim == "filter":
                res_weight = weight[assignment, :]
            elif slice_dim == "channel":
                _ori_matched_shape = list(copy.deepcopy(layer_ori_shape))
                _ori_matched_shape[1] = matched_num_filters
                trans_weight = trans_next_conv_layer_forward(weight, _ori_matched_shape)
                sliced_weight = trans_weight[assignment, :]
                res_weight = trans_next_conv_layer_backward(sliced_weight, layer_ori_shape)
        elif weight_type == "bias":
            res_weight = weight[assignment]
        elif weight_type == "first_fc_weight":
            # NOTE: please note that in this case, we pass the `estimated_shape` to `layer_ori_shape`:
            __ori_shape = weight.shape
            res_weight = weight.reshape(matched_num_filters, layer_ori_shape[2]*layer_ori_shape[3]*__ori_shape[1])[assignment, :]
            res_weight = res_weight.reshape((len(assignment)*layer_ori_shape[2]*layer_ori_shape[3], __ori_shape[1]))
        elif weight_type == "fc_weight":
            if slice_dim == "filter":
                res_weight = weight.T[assignment, :]
                #res_weight = res_weight.T
            elif slice_dim == "channel":
                res_weight = weight[assignment, :].T
        return res_weight

    # handle the conv layers part which is not changing
    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
        if (param_idx in __non_loading_indices) and (freezing_index[0] != n_layers):
            # we need to reconstruct the weights here s.t.
            # i) shapes of the weights are euqal to the shapes of the weight in original model (before matching)
            # ii) each neuron comes from the corresponding global neuron
            _matched_weight = weights[param_idx]
            _matched_num_filters = weights[__non_loading_indices[0]].shape[0]
            #
            # we now use this `_slice_dim` for both conv layers and fc layers
            if __non_loading_indices.index(param_idx) != 2:
                _slice_dim = "filter" # please note that for biases, it doesn't really matter if we're going to use filter or channel
            else:
                _slice_dim = "channel"

            if "conv" in key_name or "features" in key_name:
                if "weight" in key_name:
                    _res_weight = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
                                                        layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
                                                        weight_type="conv_weight", slice_dim=_slice_dim)
                    temp_dict = {key_name: torch.from_numpy(_res_weight.reshape(param.size()))}
                elif "bias" in key_name:
                    _res_bias = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
                                                        layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
                                                        weight_type="bias", slice_dim=_slice_dim)                   
                    temp_dict = {key_name: torch.from_numpy(_res_bias)}
            elif "fc" in key_name or "classifier" in key_name:
                if "weight" in key_name:
                    if freezing_index[0] != 6:
                        _res_weight = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
                                                            layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
                                                            weight_type="fc_weight", slice_dim=_slice_dim)
                        temp_dict = {key_name: torch.from_numpy(_res_weight)}
                    else:
                        # that's for handling the first fc layer that is connected to the conv blocks
                        _res_weight = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
                                                            layer_ori_shape=estimated_output.size(), matched_num_filters=_matched_num_filters,
                                                            weight_type="first_fc_weight", slice_dim=_slice_dim)
                        temp_dict = {key_name: torch.from_numpy(_res_weight.T)}            
                elif "bias" in key_name:
                    _res_bias = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments, 
                                                        layer_ori_shape=param.size(), matched_num_filters=_matched_num_filters,
                                                        weight_type="bias", slice_dim=_slice_dim)
                    temp_dict = {key_name: torch.from_numpy(_res_bias)}
        else:
            if "conv" in key_name or "features" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx].reshape(param.size()))}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
            elif "fc" in key_name or "classifier" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx].T)}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
        
        new_state_dict.update(temp_dict)
    matched_cnn.load_state_dict(new_state_dict)

    matched_cnn.to(device).train()
    return matched_cnn


#@title PFNM CNN code

def pdm_prepare_freq(cls_freqs, n_classes):
    freqs = []

    for net_i in sorted(cls_freqs.keys()):
        net_freqs = [0] * n_classes

        for cls_i in cls_freqs[net_i]:
            net_freqs[cls_i] = cls_freqs[net_i][cls_i]

        freqs.append(np.array(net_freqs))

    return freqs




def pdm_prepare_full_weights_cnn(nets, device="cpu"):
    """
    we extract all weights of the conv nets out here:
    """
    weights = []
    for net_i, net in enumerate(nets):
        net_weights = []
        statedict = net.state_dict()

        for param_id, (k, v) in enumerate(statedict.items()):
            if device == "cpu":
                if 'fc' in k or 'classifier' in k:
                    if 'weight' in k:
                        net_weights.append(v.numpy().T)
                    else:
                        net_weights.append(v.numpy())
                elif 'conv' in k or 'features' in k:
                    if 'weight' in k:
                        _weight_shape = v.size()
                        if len(_weight_shape) == 4:
                            net_weights.append(v.numpy().reshape(_weight_shape[0], _weight_shape[1]*_weight_shape[2]*_weight_shape[3]))
                        else:
                            pass
                    else:
                        net_weights.append(v.numpy())
            else:
                if 'fc' in k or 'classifier' in k:
                    if 'weight' in k:
                        net_weights.append(v.cpu().numpy().T)
                    else:
                        net_weights.append(v.cpu().numpy())
                elif 'conv' in k or 'features' in k:
                    if 'weight' in k:
                        _weight_shape = v.size()
                        if len(_weight_shape) == 4:
                            net_weights.append(v.cpu().numpy().reshape(_weight_shape[0], _weight_shape[1]*_weight_shape[2]*_weight_shape[3]))
                        else:
                            pass
                    else:
                        net_weights.append(v.cpu().numpy())
        weights.append(net_weights)
    return weights



#@title PFNM CNN code



def compute_full_cnn(models, weights, dataset_train_global, dataset_test_global, n_classes, device, args, args_global):
    """Note that we only handle the FC weights for now"""

    
    

    if args.model == "lenet":
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        kernel_size = 5
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1]]

        output_dim = weights[-1].shape[0]

        logger.info("Num filters: {}, Input dim: {}, hidden_dims: {}, output_dim: {}".format(num_filters, input_dim, hidden_dims, output_dim))

        matched_cnn = LeNetContainer(
                                    num_filters=num_filters,
                                    kernel_size=kernel_size,
                                    input_dim=input_dim,
                                    hidden_dims=hidden_dims,
                                    output_dim=output_dim)
    elif args.model == "vgg":
        matched_shapes = [w.shape for w in weights]
        matched_cnn = matched_vgg11(matched_shapes=matched_shapes)
    elif args.model == "simple-cnn":
        if args.dataset in ("cifar10", "cinic10"):
            input_channel = 3
        elif args.dataset == "mnist":
            input_channel = 1
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1], weights[6].shape[1]]
        matched_cnn = SimpleCNNContainer(input_channel=input_channel, 
                                        num_filters=num_filters, 
                                        kernel_size=5, 
                                        input_dim=input_dim, 
                                        hidden_dims=hidden_dims, 
                                        output_dim=10)

    elif args.model == "simple-cnn-cifar":
        if args.dataset in ("cifar10", "cinic10"):
            input_channel = 3
        elif args.dataset == "mnist":
            input_channel = 1
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1], weights[6].shape[1]]
        matched_cnn = SimpleCNNCIFARContainer(input_channel=input_channel, 
                                        num_filters=num_filters, 
                                        kernel_size=5, 
                                        input_dim=input_dim, 
                                        hidden_dims=hidden_dims, 
                                        output_dim=10)


    elif args.model == "moderate-cnn":

        num_filters = [weights[0].shape[0], weights[2].shape[0], weights[4].shape[0], weights[6].shape[0], weights[8].shape[0], weights[10].shape[0]]
        input_dim = weights[12].shape[0]
        hidden_dims = [weights[12].shape[1], weights[14].shape[1]]
        if args.dataset in ("cifar10", "cinic10"):
            matched_cnn = ModerateCNNContainer(3,
                                                num_filters, 
                                                kernel_size=3, 
                                                input_dim=input_dim, 
                                                hidden_dims=hidden_dims, 
                                                output_dim=args.n_c)
        elif args.dataset == "mnist":
            matched_cnn = ModerateCNNContainer(1,
                                                num_filters, 
                                                kernel_size=3, 
                                                input_dim=input_dim, 
                                                hidden_dims=hidden_dims, 
                                                output_dim=args.n_c)

    #logger.info("Keys of layers of convblock ...")
    new_state_dict = {}
    model_counter = 0

    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):

        if "conv" in key_name or "features" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].reshape(param.size()))}
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
        elif "fc" in key_name or "classifier" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].T)}
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}

        new_state_dict.update(temp_dict)
    matched_cnn.load_state_dict(new_state_dict)

    matched_cnn.to(device)


    return matched_cnn





def oneshot_matching(nets_list, model_meta_data, layer_type, net_dataidx_map, 
                            averaging_weights, args, num_c,
                            device="cpu"):
    # starting the neural matching
    models = nets_list
    cls_freqs = net_dataidx_map
    n_classes = num_c
    it=5
    sigma=args.sigma
    sigma0=args.sigma0
    gamma=args.gamma
    assignments_list = []
    
    batch_weights = pdm_prepare_full_weights_cnn(models, device=args.device)
    raw_batch_weights = copy.deepcopy(batch_weights)
    
    logging.info("=="*15)
    logging.info("Weights shapes: {}".format([bw.shape for bw in batch_weights[0]]))

    batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)
    res = {}
    best_test_acc, best_train_acc, best_weights, best_sigma, best_gamma, best_sigma0 = -1, -1, None, -1, -1, -1

    # gamma = 50
    # sigma = 0.1
    # sigma0 = 1.0

    n_layers = int(len(batch_weights[0]) / 2)
    num_workers = len(nets_list)
    matching_shapes = []

    first_fc_index = None

    for layer_index in range(1, n_layers):
        layer_hungarian_weights, assignment, L_next = layer_wise_group_descent(
             batch_weights=batch_weights, 
             layer_index=layer_index,
             sigma0_layers=sigma0, 
             sigma_layers=sigma, 
             batch_frequencies=batch_freqs, 
             it=it, 
             gamma_layers=gamma, 
             model_meta_data=model_meta_data,
             model_layer_type=layer_type,
             n_layers=n_layers,
             matching_shapes=matching_shapes,
             args=args
             )
        assignments_list.append(assignment)
        
        # iii) load weights to the model and train the whole thing
        type_of_patched_layer = layer_type[2 * (layer_index + 1) - 2]
        if 'conv' in type_of_patched_layer or 'features' in type_of_patched_layer:
            l_type = "conv"
        elif 'fc' in type_of_patched_layer or 'classifier' in type_of_patched_layer:
            l_type = "fc"

        type_of_this_layer = layer_type[2 * layer_index - 2]
        type_of_prev_layer = layer_type[2 * layer_index - 2 - 2]
        first_fc_identifier = (('fc' in type_of_this_layer or 'classifier' in type_of_this_layer) and ('conv' in type_of_prev_layer or 'features' in type_of_this_layer))
        
        if first_fc_identifier:
            first_fc_index = layer_index
        
        matching_shapes.append(L_next)
        #tempt_weights = [batch_weights[0][i] for i in range(2 * layer_index - 2)] + [copy.deepcopy(layer_hungarian_weights) for _ in range(num_workers)]
        tempt_weights =  [([batch_weights[w][i] for i in range(2 * layer_index - 2)] + copy.deepcopy(layer_hungarian_weights)) for w in range(num_workers)]

        # i) permutate the next layer wrt matching result
        for worker_index in range(num_workers):
            if first_fc_index is None:
                if l_type == "conv":
                    patched_weight = block_patching(batch_weights[worker_index][2 * (layer_index + 1) - 2], 
                                        L_next, assignment[worker_index], 
                                        layer_index+1, model_meta_data,
                                        matching_shapes=matching_shapes, layer_type=l_type,
                                        dataset=args.dataset, network_name=args.model)
                elif l_type == "fc":
                    patched_weight = block_patching(batch_weights[worker_index][2 * (layer_index + 1) - 2].T, 
                                        L_next, assignment[worker_index], 
                                        layer_index+1, model_meta_data,
                                        matching_shapes=matching_shapes, layer_type=l_type,
                                        dataset=args.dataset, network_name=args.model).T

            elif layer_index >= first_fc_index:
                patched_weight = patch_weights(batch_weights[worker_index][2 * (layer_index + 1) - 2].T, L_next, assignment[worker_index]).T

            tempt_weights[worker_index].append(patched_weight)

        # ii) prepare the whole network weights
        for worker_index in range(num_workers):
            for lid in range(2 * (layer_index + 1) - 1, len(batch_weights[0])):
                tempt_weights[worker_index].append(batch_weights[worker_index][lid])

        retrained_nets = []
        for worker_index in range(num_workers):
            dataidxs = net_dataidx_map[worker_index]

            logger.info("Re-training on local worker: {}, starting from layer: {}".format(worker_index, 2 * (layer_index + 1) - 2))
            retrained_cnn = local_retrain_dummy(tempt_weights[worker_index], args, 
                                            freezing_index=(2 * (layer_index + 1) - 2), device=args.device)
            retrained_nets.append(retrained_cnn)
        batch_weights = pdm_prepare_full_weights_cnn(retrained_nets, device=args.device)

    ## we handle the last layer carefully here ...
    ## averaging the last layer
    matched_weights = []
    num_layers = len(batch_weights[0])

    # with open('./matching_weights_cache/matched_layerwise_weights', 'wb') as weights_file:
    #     pickle.dump(batch_weights, weights_file)

    last_layer_weights_collector = []

    for i in range(num_workers):
        # firstly we combine last layer's weight and bias
        bias_shape = batch_weights[i][-1].shape
        last_layer_bias = batch_weights[i][-1].reshape((1, bias_shape[0]))
        last_layer_weights = np.concatenate((batch_weights[i][-2], last_layer_bias), axis=0)
        
        # the directed normalization doesn't work well, let's try weighted averaging
        last_layer_weights_collector.append(last_layer_weights)

    last_layer_weights_collector = np.array(last_layer_weights_collector)
    
    avg_last_layer_weight = np.zeros(last_layer_weights_collector[0].shape, dtype=np.float32)

    for i in range(n_classes):
        avg_weight_collector = np.zeros(last_layer_weights_collector[0][:, 0].shape, dtype=np.float32)
        for j in range(num_workers):
            avg_weight_collector += averaging_weights[j][i]*last_layer_weights_collector[j][:, i]
        avg_last_layer_weight[:, i] = avg_weight_collector

    #avg_last_layer_weight = np.mean(last_layer_weights_collector, axis=0)
    for i in range(num_layers):
        if i < (num_layers - 2):
            matched_weights.append(batch_weights[0][i])

    matched_weights.append(avg_last_layer_weight[0:-1, :])
    matched_weights.append(avg_last_layer_weight[-1, :])




    return matched_weights, assignments_list




