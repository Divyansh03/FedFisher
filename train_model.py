import copy
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
from nngeometry.metrics import FIM
from nngeometry.object import PMatKFAC, PMatDiag, PVector

class LocalUpdate(object):
    def __init__(self, args, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])

    def train_and_compute_fisher(self, net, n_c):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args['eta'], momentum = 0.9)
        step_count = 0

        for epoch in range(self.args['local_epochs']):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args['device']), labels.to(self.args['device'])
                if(self.args['augmentation']==True):
                    images = self.transform_train(images)
                optimizer.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            print ("Epoch No. ", epoch, "Loss " , sum(batch_loss)/len(batch_loss))
            
        F_kfac = FIM(model=net,
                          loader=self.ldr_train,
                          representation=PMatKFAC,
                          device='cuda',
                          n_output=n_c)
        
        F_diag = FIM(model=net,
                          loader=self.ldr_train,
                          representation=PMatDiag,
                          device='cuda',
                          n_output=n_c)

        F_diag = F_diag.get_diag()
        vec_curr = parameters_to_vector(net.parameters())            
        return vec_curr, net, F_kfac, F_diag

