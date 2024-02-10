import numpy as np
import random
import copy
import argparse
import csv
import torch
from torchvision import datasets, transforms
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset, TensorDataset

from data import get_dataset
from models import get_model
from train_model import LocalUpdate
from run_one_shot_algs import get_one_shot_model
from utils.compute_accuracy import test_img





parser = argparse.ArgumentParser()


parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--algs_to_run', nargs = '+', type=str, required=True)
parser.add_argument('--seed', type=int, required=False, default = 0)
parser.add_argument('--alpha', type = float, required = False, default = 0.1)
parser.add_argument('--num_clients', type = int, required = False, default = 5)
parser.add_argument('--num_rounds', type = int, required = False, default = 1)
parser.add_argument('--local_epochs', type=int, required= False, default = 30)
parser.add_argument('--use_pretrained', type=bool, required = False, default = False) 


args_parser = parser.parse_args()

seed = args_parser.seed
dataset = args_parser.dataset
model_name = args_parser.model
algs_to_run = args_parser.algs_to_run
local_epochs = args_parser.local_epochs
use_pretrained = args_parser.use_pretrained
alpha = args_parser.alpha
num_clients = args_parser.num_clients
num_rounds = args_parser.num_rounds
print_every_test = 1
print_every_train = 1



filename = "one_shot_results_"+str(seed)+"_"+dataset+"_"+model_name+"_"+"_"+str(local_epochs)
filename_csv = filename + ".csv"


if(dataset=='CIFAR100'):
  n_c = 100
elif (dataset == 'GTSRB'):
  n_c = 43
else: n_c = 10

dict_results = {}


for alg in algs_to_run:
  print ("Running algorithm", alg)
  print ("Using pre-trained model:", use_pretrained)

  np.random.seed(3)
  dataset_train, dataset_train_global, dataset_test_global, net_cls_counts = get_dataset(dataset, num_clients, n_c, alpha, False)
  test_loader = DataLoader(dataset_test_global, batch_size=len(dataset_test_global))

  ind = np.random.choice(len(dataset_train_global), 500)
  dataset_val = torch.utils.data.Subset(dataset_train_global, ind)

  ### Default parameters
  args={
  "bs":64,
  "local_epochs":local_epochs,
  "device":'cuda',
  "rounds":num_rounds, 
  "num_clients": num_clients,
  "augmentation": False,
  "eta": 0.01,
  "dataset":dataset,
  "model":model_name,
  "use_pretrained":use_pretrained,
  "n_c":n_c
  }



  torch.manual_seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True
  net_glob_org = get_model(args['model'], n_c, bias = False, use_pretrained = use_pretrained).to(args['device'])


  n = len(dataset_train)
  print ("No. of clients", n)

  ### Computing weights of the local models proportional to datasize
  p = np.zeros((n))
  for i in range(n):
    p[i] = len(dataset_train[i])
  p = p/np.sum(p)


  local_model_accs = []
  local_model_loss = []
  d = parameters_to_vector(net_glob_org.parameters()).numel()
  net_glob = copy.deepcopy(net_glob_org)
  initial_vector = parameters_to_vector(net_glob.parameters())


  for t in range(0,args['rounds']):

    if(dataset=='CIFAR10' or dataset=='CIFAR100' or dataset == 'CINIC10' or dataset == 'GTSRB'):
      args['augmentation'] = True

    if(use_pretrained == True):
        args['eta'] = 0.001     ### Smaller learning rate if using pretrained model
          
    ind = [i for i in range(n)]
    F_kfac_list = []
    F_diag_list = []
    model_vectors = []
    models = []
    
    for i in ind:

        print ("Training Local Model ", i)
        net_glob.train()
        local = LocalUpdate(args=args, dataset=dataset_train[i])
        model_vector, model, F_kfac, F_diag = local.train_and_compute_fisher(copy.deepcopy(net_glob), args['n_c'])
        model_vectors.append(model_vector)
        models.append(model)
        test_acc, test_loss = test_img(model, dataset_test_global,args)
        print ("Local Model ", i, "Test Acc. ", test_acc, "Test Loss ", test_loss)
        local_model_accs.append(test_acc.flatten()[0])
        local_model_loss.append(test_loss)

        ### Multiplying Fisher information with datasize weights
        F_diag = F_diag*p[i]
        for layer_id in F_kfac.data.keys():
          F_kfac.data[layer_id] = list(F_kfac.data[layer_id])

          F_kfac.data[layer_id][0].mul_(p[i])
          F_kfac.data[layer_id][1].mul_(p[i])
        F_kfac_list.append(F_kfac)
        F_diag_list.append(F_diag)

  
  dict_results['local_model_test_accuracies_'+str(alpha)+"_"+str(t)] = local_model_accs
  dict_results['local_model_test_losses_'+str(alpha)+"_"+str(t)] = local_model_loss

  ### Creating one-shot model depending on the algorithm
  net_glob = get_one_shot_model(alg, d,n,p,args,net_glob, models, model_vectors, \
  F_kfac_list, F_diag_list, dataset_val, dataset_train, dataset_train_global, \
  dataset_test_global, filename, net_cls_counts)
  
  test_acc, test_loss = test_img(net_glob, dataset_test_global,args)
  print ("Test Acc. ", test_acc, "Test Loss", test_loss)
  dict_results[alg + '_test_loss_'+str(seed)+"_"+str(t)] = test_loss
  dict_results[alg + '_test_acc_' +str(seed)+"_"+str(t)] = test_acc
  
  with open(filename_csv, 'w') as csv_file:    
    writer = csv.writer(csv_file)
    for i in dict_results.keys():
        writer.writerow([i, dict_results[i]])




with open(filename_csv, 'w') as csv_file:    
  writer = csv.writer(csv_file)
  for i in dict_results.keys():
      writer.writerow([i, dict_results[i]])
