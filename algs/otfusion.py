### Code adopted from https://github.com/sidak/otfusion

import copy
import numpy as np
import ot
import torch
from torchvision import datasets, transforms
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def isnan(x):
    return x != x

class GroundMetric:
    """
        Ground Metric object for Wasserstein computations:
    """

    def __init__(self, params, not_squared = False):
        self.params = params
        self.ground_metric_type = params['ground-metric']
        self.ground_metric_normalize = params['ground-metric-normalize']
        self.reg = params['reg']
        if params['not_squared']:
            self.squared = not params['not_squared']
        else:
            self.squared = not not_squared

        self.squared = True
        self.mem_eff = params['ground-metric-eff'] 

    def _clip(self, ground_metric_matrix):
        if self.params['debug']:
            print("before clipping", ground_metric_matrix.data)

        percent_clipped = (float((ground_metric_matrix >= self.reg * self.params.clip_max).long().sum().data) \
                           / ground_metric_matrix.numel()) * 100
        print("percent_clipped is (assumes clip_min = 0) ", percent_clipped)
        setattr(self.params, 'percent_clipped', percent_clipped)
        # will keep the M' = M/reg in range clip_min and clip_max
        ground_metric_matrix.clamp_(min=self.reg * self.params.clip_min,
                                             max=self.reg * self.params.clip_max)
        if self.params['debug']:
            print("after clipping", ground_metric_matrix.data)
        return ground_metric_matrix

    def _normalize(self, ground_metric_matrix):

        if self.ground_metric_normalize == "log":
            ground_metric_matrix = torch.log1p(ground_metric_matrix)
        elif self.ground_metric_normalize == "max":
            print("Normalizing by max of ground metric and which is ", ground_metric_matrix.max())
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.max()
        elif self.ground_metric_normalize == "median":
            print("Normalizing by median of ground metric and which is ", ground_metric_matrix.median())
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.median()
        elif self.ground_metric_normalize == "mean":
            print("Normalizing by mean of ground metric and which is ", ground_metric_matrix.mean())
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.mean()
        elif self.ground_metric_normalize == "none":
            return ground_metric_matrix
        else:
            raise NotImplementedError

        return ground_metric_matrix

    def _sanity_check(self, ground_metric_matrix):
        assert not (ground_metric_matrix < 0).any()
        assert not (isnan(ground_metric_matrix).any())

    def _cost_matrix_xy(self, x, y, p=2, squared = True):
        # TODO: Use this to guarantee reproducibility of previous results and then move onto better way
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
        if not squared:
            print("dont leave off the squaring of the ground metric")
            c = c ** (1/2)
        # print(c.size())
        if self.params['dist_normalize']:
            assert NotImplementedError
        return c


    def _pairwise_distances(self, x, y=None, squared=True):
        '''
        Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        dist = torch.clamp(dist, min=0.0)

        if self.params['activation_histograms'] and self.params['dist_normalize']:
            dist = dist/self.params.act_num_samples
            print("Divide squared distances by the num samples")

        if not squared:
            print("dont leave off the squaring of the ground metric")
            dist = dist ** (1/2)

        return dist

    def _get_euclidean(self, coordinates, other_coordinates=None):
        # TODO: Replace by torch.pdist (which is said to be much more memory efficient)

        if other_coordinates is None:
            matrix = torch.norm(
                coordinates.view(coordinates.shape[0], 1, coordinates.shape[1]) \
                - coordinates, p=2, dim=2
            )
        else:
            if self.mem_eff:
                matrix = self._pairwise_distances(coordinates, other_coordinates, squared=self.squared)
            else:
                matrix = self._cost_matrix_xy(coordinates, other_coordinates, squared = self.squared)

        return matrix

    def _normed_vecs(self, vecs, eps=1e-9):
        norms = torch.norm(vecs, dim=-1, keepdim=True)
        print("stats of vecs are: mean {}, min {}, max {}, std {}".format(
            norms.mean(), norms.min(), norms.max(), norms.std()
        ))
        return vecs / (norms + eps)

    def _get_cosine(self, coordinates, other_coordinates=None):
        if other_coordinates is None:
            matrix = coordinates / torch.norm(coordinates, dim=1, keepdim=True)
            matrix = 1 - matrix @ matrix.t()
        else:
            matrix = 1 - torch.div(
                coordinates @ other_coordinates.t(),
                torch.norm(coordinates, dim=1).view(-1, 1) @ torch.norm(other_coordinates, dim=1).view(1, -1)
            )
        return matrix.clamp_(min=0)

    def _get_angular(self, coordinates, other_coordinates=None):
        pass

    def get_metric(self, coordinates, other_coordinates=None):
        get_metric_map = {
            'euclidean': self._get_euclidean,
            'cosine': self._get_cosine,
            'angular': self._get_angular,
        }
        return get_metric_map[self.ground_metric_type](coordinates, other_coordinates)

    def process(self, coordinates, other_coordinates=None):
        # print('Processing the coordinates to form ground_metric')
        if self.params['normalize-wts']:
            print("In weight mode: normalizing weights to unit norm")
            coordinates = self._normed_vecs(coordinates)
            if other_coordinates is not None:
                other_coordinates = self._normed_vecs(other_coordinates)

        ground_metric_matrix = self.get_metric(coordinates, other_coordinates)

        self._sanity_check(ground_metric_matrix)

        ground_metric_matrix = self._normalize(ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)


        if self.params['clip_gm']:
            ground_metric_matrix = self._clip(ground_metric_matrix)

        return ground_metric_matrix



def get_histogram(args, idx, cardinality, layer_name, activations=None, return_numpy = True, float64=False):
    if activations is None:
        # returns a uniform measure
        if not args['unbalanced']:
            # print("returns a uniform measure of cardinality: ", cardinality)
            return np.ones(cardinality)/cardinality
        else:
            return np.ones(cardinality)
    else:
        # return softmax over the activations raised to a temperature
        # layer_name is like 'fc1.weight', while activations only contains 'fc1'
        print(activations[idx].keys())
        unnormalized_weights = activations[idx][layer_name.split('.')[0]]
        print("For layer {},  shape of unnormalized weights is ".format(layer_name), unnormalized_weights.shape)
        unnormalized_weights = unnormalized_weights.squeeze()
        assert unnormalized_weights.shape[0] == cardinality

        if return_numpy:
            if float64:
                return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0).data.cpu().numpy().astype(
                    np.float64)
            else:
                return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0).data.cpu().numpy()
        else:
            return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0)




def get_wassersteinized_layers_modularized(args, networks, activations=None, eps=1e-7, test_loader=None):
    '''
    Two neural networks that have to be averaged in geometric manner (i.e. layerwise).
    The 1st network is aligned with respect to the other via wasserstein distance.
    Also this assumes that all the layers are either fully connected or convolutional *(with no bias)*
    :param networks: list of networks
    :param activations: If not None, use it to build the activation histograms.
    Otherwise assumes uniform distribution over neurons in a layer.
    :return: list of layer weights 'wassersteinized'
    '''

    avg_aligned_layers = []

    T_var = None
    if args['handle_skips']:
        skip_T_var = None
        skip_T_var_idx = -1
        residual_T_var = None
        residual_T_var_idx = -1

    previous_layer_shape = None
    ground_metric_object = GroundMetric(args)

    if(args['eval_aligned']):
      model0_aligned_layers = []


    if args['gpu_id']==-1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args['gpu_id']))


    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))

    for idx, ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) in \
            enumerate(zip(networks[0].named_parameters(), networks[1].named_parameters())):

        assert fc_layer0_weight.shape == fc_layer1_weight.shape
        # print("Previous layer shape is ", previous_layer_shape)
        previous_layer_shape = fc_layer1_weight.shape

        mu_cardinality = fc_layer0_weight.shape[0]
        nu_cardinality = fc_layer1_weight.shape[0]

        layer_shape = fc_layer0_weight.shape
        layer0_shape = fc_layer0_weight.shape


        if len(layer_shape) > 2:
            is_conv = True
            # For convolutional layers, it is (#out_channels, #in_channels, height, width)
            fc_layer0_weight_data = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)
            fc_layer1_weight_data = fc_layer1_weight.data.view(fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1)
        else:
            is_conv = False
            fc_layer0_weight_data = fc_layer0_weight.data
            fc_layer1_weight_data = fc_layer1_weight.data

        if idx == 0:
            if is_conv:
                M = ground_metric_object.process(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                                fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
            else:

                M = ground_metric_object.process(fc_layer0_weight_data, fc_layer1_weight_data)

            aligned_wt = fc_layer0_weight_data
        else:

            # print("shape of layer: model 0", fc_layer0_weight_data.shape)
            # print("shape of layer: model 1", fc_layer1_weight_data.shape)
            # print("shape of previous transport map", T_var.shape)


            if is_conv:
                if args['handle_skips']:
                    assert len(layer0_shape) == 4
                    # save skip_level transport map if there is block ahead
                    if layer0_shape[1] != layer0_shape[0]:
                        if not (layer0_shape[2] == 1 and layer0_shape[3] == 1):
                            print(f'saved skip T_var at layer {idx} with shape {layer0_shape}')
                            skip_T_var = T_var.clone()
                            skip_T_var_idx = idx
                        else:
                            print(
                                f'utilizing skip T_var saved from layer layer {skip_T_var_idx} with shape {skip_T_var.shape}')
                            # if it's a shortcut (128, 64, 1, 1)
                            residual_T_var = T_var.clone()
                            residual_T_var_idx = idx  # use this after the skip
                            T_var = skip_T_var
                        print("shape of previous transport map now is", T_var.shape)
                    else:
                        if residual_T_var is not None and (residual_T_var_idx == (idx - 1)):
                            T_var = (T_var + residual_T_var) / 2
                            print("averaging multiple T_var's")
                        else:
                            print("doing nothing for skips")
                T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)
                aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)

                M = ground_metric_object.process(
                    aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                    fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
                )
            else:
                if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                    # Handles the switch from convolutional layers to fc layers
                    fc_layer0_unflattened = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var.shape[0], -1).permute(2, 0, 1)
                    aligned_wt = torch.bmm(
                        fc_layer0_unflattened,
                        T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                    ).permute(1, 2, 0)
                    aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                else:
                    aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)
                M = ground_metric_object.process(aligned_wt, fc_layer1_weight)

            # if args['skip_last_layer'] and idx == (num_layers - 1):
            #     print("Simple averaging of last layer weights. NO transport map needs to be computed")
            #     if args['ensemble_step'] != 0.5:
            #         avg_aligned_layers.append((1 - args.ensemble_step) * aligned_wt +
            #                               args.ensemble_step * fc_layer1_weight)
            #     else:
            #         avg_aligned_layers.append((aligned_wt + fc_layer1_weight)/2)
            #     return avg_aligned_layers

        mu = get_histogram(args, 0, mu_cardinality, layer0_name)
        nu = get_histogram(args, 1, nu_cardinality, layer1_name)

        cpuM = M.data.cpu().numpy()

        # print (cpuM)

        # print (mu)
        # print (nu)

        if args['exact']:
            T = ot.emd(mu, nu, cpuM)
        else:
            T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=args['reg'])


        if args['gpu_id']!=-1:
            T_var = torch.from_numpy(T).cuda(args['gpu_id']).float()
        else:
            T_var = torch.from_numpy(T).float()

    

        if args['correction']:
            if not args['proper_marginals']:
                # think of it as m x 1, scaling weights for m linear combinations of points in X
                if args['gpu_id'] != -1:
                    marginals = torch.ones(T_var.shape[0]).cuda(args['gpu_id']) / T_var.shape[0]
                else:
                    marginals = torch.ones(T_var.shape[0]) / T_var.shape[0]
                marginals = torch.diag(1.0/(marginals + eps))  # take inverse
                T_var = torch.matmul(T_var, marginals)
            else:
                marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)

                marginals = (1 / (marginals_beta + eps))

                T_var = T_var * marginals
                # print(T_var.sum(dim=0))

        if args['past_correction']:
            t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
        else:
            t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))


        # geometric_fc = ((1-args['ensemble_step']) * t_fc0_model +
        #                     args['ensemble_step'] * fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))

        if args['eval_aligned']:
            if is_conv and layer_shape != t_fc0_model.shape:
                t_fc0_model = t_fc0_model.view(layer_shape)
            model0_aligned_layers.append(t_fc0_model)


    layer_idx = 0
    model = copy.deepcopy(networks[0])

    model_state_dict = model.state_dict()


    for key, value in model_state_dict.items():

        model_state_dict[key] = model0_aligned_layers[layer_idx]
        layer_idx += 1


    model.load_state_dict(model_state_dict)

    return model


def one_shot_otfusion(net_glob, models, p, args_ot):
  
    pivot = args_ot['pivot']
    n = len(models)
    model_avg_vector = p[pivot]*parameters_to_vector(models[pivot].parameters())

    for i in range(n):
      if(i!=pivot):
        model = get_wassersteinized_layers_modularized(args_ot,[models[i],models[pivot]])
        print ("Aligned Model ", i)
        model_avg_vector += p[i]*parameters_to_vector(model.parameters())

    
    net_glob_copy = copy.deepcopy(net_glob)
    vector_to_parameters(model_avg_vector, net_glob_copy.parameters())

    return net_glob_copy
    

