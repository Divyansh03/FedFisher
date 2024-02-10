import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset, DataLoader


def __getDirichletData__(y, n, alpha, num_c):

        min_size = 0
        N = len(y)
        net_dataidx_map = {}
        p_client = np.zeros((n,num_c))

        for i in range(n):
          p_client[i] = np.random.dirichlet(np.repeat(alpha,num_c))
        idx_batch = [[] for _ in range(n)]

        for k in range(num_c):
            idx_k = np.where(y == k)[0]
            np.random.shuffle(idx_k)
            proportions = p_client[:,k]
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

        for j in range(n):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        net_cls_counts = {}

        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(y[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp

        local_sizes = []
        for i in range(n):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        weights = local_sizes / np.sum(local_sizes)

        print('Data statistics: %s' % str(net_cls_counts))
        print('Data ratio: %s' % str(weights))

        return idx_batch, net_cls_counts

def get_dataset(datatype, n_client, n_c, alpha, partition_equal=True):

    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])
    trans_fashionmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trans_svhn = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    if(datatype=='CIFAR10' or datatype=='SVHN' or datatype == 'GTSRB' or datatype=='CIFAR100' or datatype =='FashionMNIST' or datatype == 'CINIC10'):
    
        if(datatype=='CIFAR10'):
            dataset_train_global = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=trans_cifar)
            dataset_test_global = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=trans_cifar)

        if(datatype=='SVHN'):
            dataset_train_global = datasets.SVHN('./data/svhn', split="train",download=True, transform=transforms.Compose([transforms.ToTensor()]))
            dataset_test_global = datasets.SVHN('./data/svhn', split="test",download = True, transform=transforms.Compose([transforms.ToTensor()]))

        if(datatype == 'GTSRB'):
            transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32)), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            dataset_train_global = datasets.GTSRB('./data/gtsrb', split="train",download=True, transform=transform)
            dataset_test_global = datasets.GTSRB('./data/gtsrb', split="test",download = True, transform=transform)

        if(datatype=='CINIC10'):
            cinic_mean = [0.47889522, 0.47227842, 0.43047404]
            cinic_std = [0.24205776, 0.23828046, 0.25874835]
            dataset_train_global = datasets.ImageFolder('./data/cinic_train',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=cinic_mean,std=cinic_std)]))
            dataset_test_global = datasets.ImageFolder('./data/cinic_test',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=cinic_mean,std=cinic_std)]))

        elif(datatype=='CIFAR100'):
            dataset_train_global = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans_cifar)
            dataset_test_global = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=trans_cifar)

        elif(datatype=='FashionMNIST'):
            dataset_train_global = datasets.FashionMNIST('./data/fashionmnist', train=True, download=True, transform=trans_fashionmnist)
            dataset_test_global = datasets.FashionMNIST('./data/fashionmnist', train=False, download=True, transform=trans_fashionmnist)

        train_loader = DataLoader(dataset_train_global, batch_size=len(dataset_train_global))
        test_loader  = DataLoader(dataset_test_global, batch_size=len(dataset_test_global))
        X_train = next(iter(train_loader))[0].numpy()
        Y_train = next(iter(train_loader))[1].numpy()
        inds, net_cls_counts = __getDirichletData__(Y_train, n_client, alpha, n_c)
        dataset_train=[]
        for (i,ind) in enumerate(inds):

            ind = inds[i]
            x = X_train[ind]
            y = Y_train[ind]
            x_train = torch.Tensor(x)
            y_train = torch.LongTensor(y)

            print ("Client ", i, " Training examples: " , len(x_train))
            dataset_train_torch = TensorDataset(x_train,y_train)
            dataset_train.append(dataset_train_torch)
    
    return dataset_train, dataset_train_global, dataset_test_global, net_cls_counts
    







    

    







