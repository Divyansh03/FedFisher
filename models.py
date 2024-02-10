import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, bias = False, n_out = 10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, bias = bias)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, bias = bias)
        self.fc1 = nn.Linear(256, 120, bias = bias)
        self.fc2 = nn.Linear(120, 84, bias = bias)
        self.fc3 = nn.Linear(84, n_out, bias = bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FedNet(nn.Module):
    def __init__(self, bias = False, n_out=10):
        super(FedNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, bias=bias)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, bias=bias)
        self.fc1 = nn.Linear(64*5*5, 512, bias=bias)
        self.fc2 = nn.Linear(512, 128, bias=bias)
        self.fc3 = nn.Linear(128, n_out, bias=bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




class BasicBlock_mod(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_mod, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias = False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias = False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = F.relu(out + self.shortcut(x))
        return out

class ResNet_mod(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_mod, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=1, bias = False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, 1000, bias = False)
        self.linear = nn.Linear(1000, 200, bias = False)
        self.final = nn.Linear(200, num_classes, bias = False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.linear(out)
        out = self.final(out)
        return out


def ResNet18_mod(n_c):
    return ResNet_mod(BasicBlock_mod, [2,2,2,2], num_classes = n_c)


#### This architecture is used for leading the ResNet18 pre-trained on TinyImageNet
class BasicBlock_tiny(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_tiny, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias = False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias = False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = F.relu(out + self.shortcut(x))
        return out

class ResNet_tiny(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_tiny, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=1, bias = False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, 1000, bias = False)
        self.linear = nn.Linear(1000, num_classes, bias = False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.linear(out)
        return out

def ResNet18_tiny(n_c):
    return ResNet_tiny(BasicBlock_tiny, [2,2,2,2], num_classes = n_c)




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias = False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias = False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = F.relu(out + self.shortcut(x))
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias = False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes, bias = False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(n_c):
    return ResNet(BasicBlock, [2,2,2,2], num_classes = n_c)




def get_model(model,n_c, bias = False, use_pretrained = False, dataset = 'CIFAR10'):
  if(model=='ResNet18'):
    if(use_pretrained == True):
        print ("Using pretrained model")
        pre_net = ResNet18_tiny(200)
        pre_net.load_state_dict(torch.load('resnet_18_tiny_imagenet_40.pt'))
        net = ResNet18_mod(n_c)
        pre_net_dict = pre_net.state_dict()
        net_dict = net.state_dict()
        for layer in pre_net_dict:
            net_dict[layer] = pre_net_dict[layer]
        net.load_state_dict(net_dict)
    else:
         net = ResNet18(n_c)
    return net
  elif(model == 'CNN'):
    return FedNet(bias = bias, n_out = n_c)
  elif (model=="LeNet"):
    return LeNet(bias = bias, n_out = n_c)
