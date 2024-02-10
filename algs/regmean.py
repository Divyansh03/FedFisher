### Code adopted from https://github.com/bloomberg/dataless-model-merging

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import re
import copy
from utils.compress_fisher import compress_regmean_gram, quantize_layer
from utils.compute_accuracy import test_img
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def filter_params_to_merge(param_names, exclude_param_regex):
    params_to_merge = []
    for name in param_names:
        valid = not any([re.match(patt, name) for patt in exclude_param_regex])
        if valid:
            params_to_merge.append(name)
    return params_to_merge


def filter_modules_by_regex(base_module, include_patterns, include_type):
    modules = {}
    for name, module in base_module.named_modules():
        valid_name = not include_patterns or any([re.match(patt, name) for patt in include_patterns])
        valid_type = not include_type or any([isinstance(module, md_cls) for md_cls in include_type])
        if valid_type and valid_name:
            modules[name] = module
    return modules



def compute_gram(model, dataset):
    
    train_dataloader = DataLoader(dataset, batch_size = 16, shuffle=True)
    grams = {} # gram matrices for each linear layer inputs
    xn = {} # number of examples used for computing gram

    def get_gram(name):
        def hook(module, input, output):
            x = input[0].detach() # $[b,t,h]
            x = x.view(-1, x.size(-1))
            xtx = torch.matmul(x.transpose(0,1), x) # [h,h]
            if name not in grams:
                grams[name] = xtx / x.size(0)
                xn[name] = x.size(0)
            else:
                grams[name] = (grams[name] * xn[name] + xtx) / (x.size(0) + xn[name])
                xn[name] += x.size(0)
        return hook

    linear_modules = filter_modules_by_regex(model, None, [nn.Linear])
    handles = []
    for name, module in linear_modules.items():
        handle = module.register_forward_hook(get_gram(name))
        handles.append(handle)

    n_step = 1000
    total = n_step if n_step > 0 else len(train_dataloader)
    for step, (images, labels) in enumerate(train_dataloader):
    
        if n_step > 0 and step == n_step:
            break

        images = images.to('cuda')
        outputs = model(images)

    for handle in handles:
        handle.remove()

    return grams



def avg_merge(p, local_models, global_model, regmean_grams=None, a = 0.9, **kwargs):
    params = {}
    for local_model in local_models:
        n2p = {k: v for k,v in local_model.named_parameters()}
        merge_param_names = filter_params_to_merge([n for n in n2p], ['.*classifier.*']) # for glue label spaces are different
        print (merge_param_names)
        for n in merge_param_names:
            if n not in params:
                params[n] = []
            params[n].append(n2p[n])

    avg_params = regmean_merge(p, params, regmean_grams, a = a)


    return avg_params

def copy_params_to_model(avg_params, model):
    for n, p in model.named_parameters():
        if n in avg_params:
            p.data.copy_(avg_params[n])

def reduce_non_diag(cov_mat, a):
    diag_weight = torch.diag(torch.ones(cov_mat.size(0)) - a).to(cov_mat.device)
    non_diag_weight = torch.zeros_like(diag_weight).fill_(a)
    weight = diag_weight + non_diag_weight
    ret = cov_mat * weight
    return ret

def regmean_merge(p, all_params, all_grams, a = 0.9):
    avg_params = {}
    n_model = len(all_grams)
    for name in all_params:
        h_avged = False
        if name.endswith('.weight'):
            print(f'Regmean: {name}')
            module_name = name[:-len('.weight')]
            if module_name in all_grams[0]:
                gram_m_ws, grams = [], []

                for model_id, model_grams in enumerate(all_grams):
                    param_grams = p[model_id]*model_grams[module_name]
                    # for roberta we dont need this; but it is important for deberta and t5
                    param_grams = reduce_non_diag(param_grams, a= a)

                    param = all_params[name][model_id]
                    gram_m_ws.append(torch.matmul(param_grams, param.transpose(0,1)))
                    grams.append(param_grams)
                sum_gram = sum(grams)
                sum_gram_m_ws = sum(gram_m_ws)
                sum_gram_inv = torch.pinverse(sum_gram)
                wt = torch.matmul(sum_gram_inv, sum_gram_m_ws)
                w = wt.transpose(0,1)
                avg_params[name] = w
                h_avged = True
        if not h_avged: # if not averaged with regmean, then do simple avg
            avg_params[name] = torch.stack(all_params[name],0).mean(0)
           
    return avg_params


def regmean_global_merge(p, args, net_glob, models, model_vectors, dataset_train, dataset_val):

    
    grams_list = []
    n = len(models)

    new_models_list = []

    for i in range(n):
        with torch.no_grad():
            grams = compute_gram(models[i], dataset_train[i])
        grams_list.append(grams)

    for i in range(n):
        model_new = copy.deepcopy(models[i])
        model_vector_quantized = quantize_layer(model_vectors[i],model_new,15)
        vector_to_parameters(model_vector_quantized, model_new.parameters())
        new_models_list.append(model_new)

    compress_regmean_gram(grams_list,models[0])
    a_range = [0.1, 0.9]
    best_params = None
    best_acc = 0
    for a in a_range:
      regmean_avg_params = avg_merge(p, new_models_list, new_models_list[0], regmean_grams=grams_list, a = a)
      regmean_model = copy.deepcopy(net_glob)
      regmean_model.load_state_dict(regmean_avg_params)
      val_test_acc, val_test_loss = test_img(regmean_model, dataset_val, args)
      print ("Val Test Acc: ", val_test_acc, "Val Test Loss", val_test_loss)
      if(val_test_acc > best_acc):
        best_acc = val_test_acc
        best_params = regmean_avg_params


    return best_params
    

    

    
