import numpy as np
import copy
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters



from data import get_dataset
from models import get_model
from train_model import LocalUpdate
from utils.compute_accuracy import test_img
from utils.compress_fisher import quantize,compress_fisher_kfac, quantize_layer
from algs.fisher_avg import one_shot_fisher_kfac, one_shot_fisher_diag
from algs.otfusion import one_shot_otfusion
from algs.dense import KLDiv,AdvSynthesizer,Ensemble,kd_train,Generator,test
from algs.pfnm import oneshot_matching, compute_full_cnn
from algs.regmean import regmean_global_merge



def get_fedavg_model(d, n, p, args, net_glob, model_vectors):
    fedavg_model = copy.deepcopy(net_glob)
    model_avg = torch.zeros(d).to(args['device'])
    for i in range(n):
      model_avg += p[i]*model_vectors[i]
    vector_to_parameters(model_avg,fedavg_model.parameters())
    return fedavg_model


def get_fisher_merge_model(d, n, p, args, net_glob, model_vectors, F_diag_list):
    

    grad_diag_sum = torch.zeros(d).to(args['device'])
    F_diag_sum = torch.zeros(d).to(args['device'])
    model_vector_sum = torch.zeros(d).to(args['device'])
    model_avg = torch.zeros(d).to(args['device'])

    for i in range(n):
      model_avg += p[i]*model_vectors[i]

    for i in range(n):
      F_diag_quantized = quantize_layer(F_diag_list[i],net_glob,15)
      model_vector_quantized = quantize_layer(model_vectors[i],net_glob,15)
      grad_diag = F_diag_quantized*model_vector_quantized
      grad_diag_sum += grad_diag
      F_diag_sum += F_diag_quantized
      model_vector_sum += p[i]*model_vector_quantized


    ind = torch.where(F_diag_sum < 1e-6)[0].detach().cpu().numpy()
    fisher_avg_model = copy.deepcopy(net_glob)
    fedavg_quantized_model = copy.deepcopy(net_glob)
    vector_to_parameters(model_vector_sum,fedavg_quantized_model.parameters())
    fisher_avg = grad_diag_sum/F_diag_sum
    fisher_avg[ind] = model_avg[ind]
    vector_to_parameters(fisher_avg,fisher_avg_model.parameters())
    return fisher_avg_model



def get_fisher_diag_model(d, n, p, args, net_glob, model_vectors, F_diag_list, dataset_val):

    args_fisher = {}
    args_fisher['eta'] = 0.01  
    args_fisher['T'] = 2000
    grad_diag_sum = torch.zeros(d).to(args['device'])
    F_diag_sum = torch.zeros(d).to(args['device'])
    model_vector_sum = torch.zeros(d).to(args['device'])

    for i in range(n):
      F_diag_quantized = quantize_layer(F_diag_list[i],net_glob,15)
      model_vector_quantized = quantize_layer(model_vectors[i],net_glob,15)
      grad_diag = F_diag_quantized*model_vector_quantized
      grad_diag_sum += grad_diag
      F_diag_sum += F_diag_quantized
      model_vector_sum += p[i]*model_vector_quantized


    fisher_avg_model = copy.deepcopy(net_glob)
    fedavg_quantized_model = copy.deepcopy(net_glob)
    vector_to_parameters(model_vector_sum,fedavg_quantized_model.parameters())
    fisher_avg = one_shot_fisher_diag(fedavg_quantized_model, F_diag_sum, grad_diag_sum, p, dataset_val, args_fisher, args)
    vector_to_parameters(fisher_avg,fisher_avg_model.parameters())
    return fisher_avg_model


def get_fisher_kfac_model(d, n, p, args, net_glob, models, model_vectors, F_kfac_list, dataset_val, dataset_train):
    
    args_fisher = {}
    args_fisher['eta'] = 0.01  
    args_fisher['lambda'] = 0
    args_fisher['T'] = 2000
    model_vector_sum = torch.zeros(d).to(args['device'])

    for i in range(n):
      model_vector_quantized = quantize_layer(model_vectors[i],net_glob,15)
      model_vector_sum += p[i]*model_vector_quantized

    fedavg_quantized_model = copy.deepcopy(net_glob)
    vector_to_parameters(model_vector_sum,fedavg_quantized_model.parameters())
    F_kfac_list_comp, grad_avg_comp = compress_fisher_kfac(args, F_kfac_list, models, model_vectors, dataset_train,p)
    fisher_avg_model = copy.deepcopy(net_glob)
    fisher_avg = one_shot_fisher_kfac(fedavg_quantized_model, F_kfac_list_comp, grad_avg_comp, p, dataset_val, args_fisher, args)
    vector_to_parameters(fisher_avg,fisher_avg_model.parameters())
    return fisher_avg_model

def get_otfusion_model(d, n, p, args, net_glob, models, model_vectors, dataset_val):

    args_ot= {'eval_aligned':True,
              'gpu_id':0,
              'importance':None,
              'exact':True,
              'past_correction':False,
              'correction':False,
              'clip_gm': False,
              'not_squared': True,
              'proper_marginals':False,
              'ensemble_step':0.5,
              'skip_last_layer':False,
              'handle_skips':False,
              'ground-metric': 'euclidean',
              'ground-metric-normalize': 'none',
              'reg': 1e-7,
              'dist_normalize':False,
              'activation_histograms': True,
              'debug':False,
              'geom-ensemble-type': 'wts',
              'ground-metric-eff': True,
              'normalize-wts':False,
              'unbalanced': False,
              'pivot': 0
              }

    if(args['model'] =='ResNet18'):
        args_ot['handle_skips'] = True
    

    ### Tuning correction and past_correction hyperparameter
    correction_range = [True, False]
    past_correction_range = [True, False]
    best_acc_test = 0
    best_correction = False
    best_past_correction = False

    for correction in correction_range:
        for past_correction in past_correction_range:

            args_ot['correction'] = correction
            args_ot['past_correction'] = past_correction
            otfusion_model = one_shot_otfusion(net_glob, models, p, args_ot)
            sum_acc_test, sum_loss_test = test_img(otfusion_model, dataset_val, args)
            if(sum_acc_test > best_acc_test):
                best_acc_test = sum_acc_test
                best_correction = correction
                best_past_correction = past_correction

    args_ot['correction'] = best_correction
    args_ot['past_correction'] = best_past_correction
    otfusion_model = one_shot_otfusion(net_glob, models, p, args_ot)
    return otfusion_model


def get_regmean_model(d, n, p, args, net_glob, models, model_vectors, dataset_val, dataset_train):
    regmean_avg_params = regmean_global_merge(p, args, net_glob, models, model_vectors, dataset_train, dataset_val)
    regmean_model = copy.deepcopy(net_glob)
    regmean_model.load_state_dict(regmean_avg_params)
    return regmean_model



def get_pfnm_model(d, n, p, args, net_glob, models, model_vectors, dataset_val, dataset_train_global, dataset_test_global, net_cls_counts):
      
    class Namespace():
      def __init__(self, **kwargs):
          self.__dict__.update(kwargs)

    models_dummy_bias = []


    '''
    PFNM expects models to have a bias term by default whereas our models do not have a bias term
    in order to be compatible with OTFusion. To get around this, we create dummy models with the bias 
    set to zero for PFNM
    ''' 

    for i in range(n):

      net_bias = get_model(args['model'],args['n_c'], bias = True).to(args['device'])
      state_dict_b = net_bias.state_dict()
      state_dict = models[i].state_dict()


      for key in state_dict_b.keys():
        if('bias' in key):
          state_dict_b[key] = torch.zeros(size=state_dict_b[key].shape).to(args['device'])
        else:
          state_dict_b[key] = state_dict[key]

      net_bias.load_state_dict(state_dict_b)
      models_dummy_bias.append(net_bias)
    

    model_meta_data = []
    layer_type = []

    for (k, v) in models_dummy_bias[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)


    n_classes = args['n_c']
    averaging_weights = np.zeros((n, n_classes), dtype=np.float32)


    for i in range(n_classes):
        total_num_counts = 0
        worker_class_counts = [0] * n
        for j in range(n):
            if i in net_cls_counts[j].keys():
                total_num_counts += net_cls_counts[j][i]
                worker_class_counts[j] = net_cls_counts[j][i]
            else:
                total_num_counts += 0
                worker_class_counts[j] = 0
        averaging_weights[:, i] = worker_class_counts / total_num_counts

    
    
    args_pfnm = Namespace( 
          device='cuda',
          dataset = 'cifar10',
          model = 'simple-cnn',
          sigma0 = 1,
          sigma = 1,
          n_c = args['n_c'],
          gamma = 7)

    if(args['dataset'] == 'MNIST' or args['dataset'] == 'FashionMNIST'):
      args_pfnm.dataset = 'mnist'



    hungarian_weights, assignments_list = oneshot_matching(models_dummy_bias, model_meta_data, layer_type, net_cls_counts, averaging_weights, args_pfnm, args['n_c'], device=args_pfnm.device)
    net_pfnm = compute_full_cnn(hungarian_weights,
                                    hungarian_weights,
                                    dataset_train_global,
                                    dataset_test_global,
                                    n_classes,
                                    device=args_pfnm.device,
                                    args=args_pfnm, args_global = args)
    return net_pfnm



def get_dense_model(d, n, p, args, net_glob, models, model_vectors, dataset_val, filename):
    
    args['dataset'] = args['dataset'].lower()
    save_dir_str = filename
    fedavg_model = get_fedavg_model(d, n, p, args, net_glob, model_vectors)
    global_model = copy.deepcopy(fedavg_model)
    global_model.train()

    model_list = []
    for i in range(len(models)):
        net =copy.deepcopy(net_glob)
        net.load_state_dict(models[i].state_dict())
        model_list.append(net)

    ensemble_model = Ensemble(model_list,p)
    # data generator
    nz = 256
    nc = 3 if "cifar" in args['dataset'] or args['dataset'] == "svhn" or "cinic" in args['dataset'] else 1
    img_size = 32 if "cifar" in args['dataset'] or args['dataset'] == "svhn" or "cinic" in args['dataset']  else 28
    generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=nc).cuda()
    img_size2 = (3, 32, 32) if "cifar" in args['dataset'] or args['dataset'] == "svhn" or "cinic" in args['dataset'] else (1, 28, 28)
    synthesizer = AdvSynthesizer(ensemble_model, model_list, global_model, generator,
                                  nz=nz, num_classes=args['n_c'], img_size=img_size2,
                                  iterations=30, lr_g=1e-3,
                                  synthesis_batch_size=256,
                                  sample_batch_size=256,
                                  adv=0.5, bn=1, oh=1,
                                  save_dir=save_dir_str, dataset=args['dataset'],p=p)

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    criterion = KLDiv(T=20)
    lr_dense = 0.01
    if(args['model'] == 'ResNet18' and args['use_pretrained'] == True):
        lr_dense = 0.001

    optimizer = torch.optim.SGD(global_model.parameters(), lr=lr_dense, momentum=0.9)
    global_model.train()
    distill_acc = []
    distill_loss = []
    cur_ep = 0
    sum_acc_test_best = 0
    dense_model = copy.deepcopy(global_model)

    for epoch in range(20):
        synthesizer.gen_data(cur_ep)  # g_steps
        cur_ep += 1
        kd_train(synthesizer, [global_model, ensemble_model], criterion, optimizer)  # # kd_steps
        print ("Epoch No.", epoch)
        sum_acc_test, sum_loss_test = test_img(global_model, dataset_val, args)
        if(sum_acc_test > sum_acc_test_best):
          sum_acc_test_best = sum_acc_test
          dense_model = copy.deepcopy(global_model)
        distill_acc.append(sum_acc_test)
        distill_loss.append(sum_loss_test)
        
    return dense_model


def get_one_shot_model(alg,d,n,p,args,net_glob, models, model_vectors, F_kfac_list, F_diag_list, dataset_val,dataset_train, dataset_train_global, dataset_test_global, filename, net_cls_counts):
    if(alg=='fedavg'):
        return get_fedavg_model(d, n, p, args, net_glob, model_vectors)
    elif(alg == 'fisher_merge'):
        return get_fisher_merge_model(d, n, p, args, net_glob, model_vectors, F_diag_list)
    elif(alg == 'fedfisher_diag'):
        return get_fisher_diag_model(d, n, p, args, net_glob, model_vectors, F_diag_list, dataset_val)
    elif(alg == 'fedfisher_kfac'):
        return get_fisher_kfac_model(d, n, p, args, net_glob, models, model_vectors, F_kfac_list, dataset_val, dataset_train)
    elif(alg == 'otfusion'):
        return get_otfusion_model(d, n, p, args, net_glob, models, model_vectors, dataset_val)
    elif(alg == 'regmean'):
        return get_regmean_model(d, n, p, args, net_glob, models, model_vectors, dataset_val, dataset_train)
    elif(alg == 'pfnm'):
        return get_pfnm_model(d, n, p, args, net_glob, models, model_vectors, dataset_val, dataset_train_global, dataset_test_global, net_cls_counts)
    elif(alg == 'dense'):
        return get_dense_model(d, n, p, args, net_glob, models, model_vectors, dataset_val, filename)
    return





