# FedFisher: Leveraging Fisher Information for One-Shot Federated Learning

This repository provides the code for the paper "FedFisher: Leveraging Fisher Information for One-Shot Federated Learning" by Divyansh Jhunjhunwala, Shiqiang Wang, and Gauri Joshi, published in AISTATS 2024.


# Instructions

Our results can be replicated by running the file `main.py`. The file takes the following arguments.

### Required Arguments

• `--dataset`: Choice of dataset. Possible choices are `FashionMNIST`, `SVHN`, `CIFAR10`, `CINIC10`, `CIFAR100`, and `GTSRB`. Note that for `CINIC10`, the train and test data first needs to be downloaded from https://github.com/BayesWatch/cinic-10.

• `--model`: The model to use. Possible choices are `LeNet`, `CNN`, and `ResNet18`.

• `--algs_to_run`: The one-shot algorithms to run. Note that you can specify more than one algorithm. Possible choices are `fedavg`, `otfusion`, `pfnm`, `regmean`, `dense`, `fisher_merge`, `fedfisher_diag` and `fedfisher_kfac`.

### Default arguments

• `--seed`: Seed for reproducibility. The default value is 0.

• `--alpha`: Heterogeneity parameter when splitting the dataset across clients. The default value is 0.1.

• `--num_clients`: Number of clients in the setup. The default value is 5.

• `--num_rounds`: Number of rounds of local training and aggregation. The default value is 1.

• `--local_epochs`: Number of local epochs run by clients. The default value is 30. 

• `--use_pretrained`: Whether to use a pre-trained model or not. The default value is False.

An example of a command to run `main.py` is given below:

```bash
python main.py --dataset 'FashionMNIST' --model 'LeNet' --local_epochs 30 --algs_to_run 'fedfisher_kfac' 'fedavg'              
```



# Notes

• The `pfnm` algorithm only works with `LeNet` and `CNN` models.

• The `--use_pretrained` only works when using the `ResNet18` model. We have also provided the pre-trained checkpoint `resnet_18_tiny_imagenet_40.pt` which is a ResNet18 model pre-trained on downsampled 32x32 TinyImageNet dataset. 

• We are using a modified `ResNet18` architecture without BatchNorm layers to be compatible with all algorithms.

# Requirements

Requirements can be found in the requirements.txt file.

# References

• The code for `otfusion` algorithm is adopted from https://github.com/sidak/otfusion.

• The code for `regmean` algorithm is adopted from https://github.com/bloomberg/dataless-model-merging.

• The code for `pfnm` algorithm is adopted from https://github.com/IBM/FedMA.

• The code for `dense` algorithm is adopted from https://github.com/zj-jayzhang/DENSE.

