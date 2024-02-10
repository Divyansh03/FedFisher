import copy
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from nngeometry.metrics import FIM
from nngeometry.object import PMatKFAC, PMatDiag, PVector
from utils.compute_accuracy import test_img


def get_dot_product(F_mat_list,w, p):
    n = len(F_mat_list)
    v = F_mat_list[0].mv(w)
    v = v.__rmul__(1/p[0])
    for i in range(1,n):
        x = F_mat_list[i].mv(w)
        x = x.__rmul__(1/p[i])
        v = v.__add__(x)
    return v


def element_mul(self, other):
        if self.dict_repr is not None and other.dict_repr is not None:
            v_dict = dict()
            for l_id, l in self.layer_collection.layers.items():
                if l.bias is not None:
                    v_dict[l_id] = (self.dict_repr[l_id][0]*
                                    other.dict_repr[l_id][0],
                                    self.dict_repr[l_id][1]*
                                    other.dict_repr[l_id][1])
                else:
                    v_dict[l_id] = (self.dict_repr[l_id][0]*
                                    other.dict_repr[l_id][0],)
            return PVector(self.layer_collection, dict_repr=v_dict)
        elif self.vector_repr is not None and other.vector_repr is not None:
            return PVector(self.layer_collection,
                           vector_repr=self.vector_repr+other.vector_repr)
        else:
            return PVector(self.layer_collection,
                           vector_repr=(self.get_flat_representation() +
                                        other.get_flat_representation()))
            
def inv_sqrt(self,epsilon):

        if self.dict_repr is not None:
            v_dict = dict()
            for l_id, l in self.layer_collection.layers.items():
                if l.bias:
                    v_dict[l_id] = (1/torch.sqrt(self.dict_repr[l_id][0] + epsilon),
                                    1/torch.sqrt(self.dict_repr[l_id][1] + epsilon))
                else:
                    v_dict[l_id] = (1/torch.sqrt(self.dict_repr[l_id][0] + epsilon),)
            return PVector(self.layer_collection, dict_repr=v_dict)
        else:
            return PVector(self.layer_collection,
                           vector_repr=x * self.vector_repr)



def one_shot_fisher_kfac(net_glob, F_kfac_list, grad, p, dataset_val, args_fisher, args):

    w = PVector.from_model(net_glob)
    w_avg = PVector.from_model(net_glob)
    net_glob_copy = copy.deepcopy(net_glob)
    eta = args_fisher['eta']
    T = args_fisher['T']
    test_acc_tracker = []
    test_acc_i_max = 0
    
    with torch.no_grad():
                
        mom = w.__rmul__(0)
        delta = w.__rmul__(0)

        ### Using Adam optimizer for the global optimization in FedFisher(K-FAC)
        for k in range(T):

            v = get_dot_product(F_kfac_list,w,p)
            v = v.__sub__(grad)
            mom = v.__add__(mom.__rmul__(0.9))
            delta = element_mul(v, v).__add__(delta.__rmul__(0.99))
            v = element_mul(mom, inv_sqrt(delta,0.01))
            v = v.__rmul__(eta)
            w = w.__sub__(v)

            if(k%100==0):
                w_vec_estimate = w.get_flat_representation()
                vector_to_parameters(w_vec_estimate,net_glob_copy.parameters())
                test_acc_i, test_loss_i = test_img(net_glob_copy, dataset_val, args)
                if(test_acc_i > test_acc_i_max):
                  test_acc_i_max = test_acc_i
                  best_parameters = w_vec_estimate

                print ("Val Test Acc: ", test_acc_i, " Val Test Loss: ", test_loss_i)
                test_acc_tracker.append(test_acc_i)

    
    return best_parameters


def one_shot_fisher_diag(net_glob, F_weight_sum, grad_avg, p, dataset_val, args_fisher, args):

    net_glob_copy = copy.deepcopy(net_glob)
    w_avg = parameters_to_vector(net_glob.parameters())
    w = parameters_to_vector(net_glob.parameters())
    T = args_fisher['T']
    eta = args_fisher['eta']
    test_acc_i_max = 0

    with torch.no_grad():

      test_acc_tracker = []
      test_acc_i_max = 0
      mom = 0*w
      delta = 0*w

      ### Using Adam optimizer for the global optimization in FedFisher(Diag)
      for k in range(T):
        v = F_weight_sum*w - grad_avg
        mom = v + mom*0.9
        delta = v*v + delta*0.99
        w = w - args_fisher['eta']*mom*1/(torch.sqrt(delta) + 0.01)

        if(k%100==0):

            w_vec_estimate = w
            vector_to_parameters(w_vec_estimate,net_glob_copy.parameters())
            test_acc_i, test_loss_i = test_img(net_glob_copy, dataset_val, args)
            if(test_acc_i > test_acc_i_max):
              test_acc_i_max = test_acc_i
              best_parameters = w

            print ("Val Test Acc: ", test_acc_i, " Val Test Loss: ", test_loss_i)
            test_acc_tracker.append(test_acc_i)

      
    return best_parameters
