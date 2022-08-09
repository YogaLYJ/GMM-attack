import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import math

'''
    Gaussian Neuron
'''

class GMM(nn.Module):

    def __init__(self, target_model, batch_size, S_x, S_y, mu_x, mu_y, p, pert_scale):
        super(GMM, self).__init__()

        self.target_model=target_model
        self.pert_scale=pert_scale/255

        # Parameters in one Gaussian Model
        self.S1 = nn.Parameter(torch.ones(batch_size) * 0.1, requires_grad=True)  # scale for x 0.1
        self.S2 = nn.Parameter(torch.ones(batch_size) * 0.1, requires_grad=True)  # scale for y 0.1
        self.S_x=nn.Parameter(torch.ones(batch_size)*S_x,requires_grad=True) # variance of x
        self.S_y=nn.Parameter(torch.ones(batch_size)*S_y,requires_grad=True) # variance of y
        self.mu_x=nn.Parameter(torch.ones(batch_size)*mu_x,requires_grad=True) # mean value of x
        self.mu_y=nn.Parameter(torch.ones(batch_size)*mu_y,requires_grad=True) # mean value of y
        self.R = nn.Parameter(torch.zeros(batch_size), requires_grad=True)  # rotation

        # The weight of the Gaussian Model
        self.p = nn.Parameter(torch.ones(batch_size).view(-1, 1, 1) * p, requires_grad=True)  # multiplier of Gaussian

        # Some constant values
        self.e=nn.Parameter(torch.ones(1)*2.71828,requires_grad=False)
        self.pi = nn.Parameter(torch.ones(1) * 3.1415926, requires_grad=False)

    # add perturbations to the original image
    def forward(self,x):
        batch_size,_,h,w=x.size()

        pert=self.pert_maker(batch_size,(h,w))
        pert=pert.unsqueeze(1)

        pert = pert * self.pert_scale * self.p
        x=x+pert

        x.data=torch.clamp(x.data,0,1)

        return pert, x

    def get_func(self, x, y):
        S = 10

        # rotation
        x, y = self.rotate(x, y, self.R)

        # scaling
        x, y = x * self.S1.view(-1, 1, 1) * S, y * self.S2.view(-1, 1, 1) * S

        index = (x - self.mu_x) ** 2 / (self.S_x ** 2 + 1e-6) + (y - self.mu_y) ** 2 / (self.S_y ** 2 + 1e-6)
        ratio = torch.abs(1 / ((2 * self.pi * self.S_x * self.S_y) + 1e-6))

        return ratio * self.e.pow(- index / 2)

    def pert_maker(self,batch_size,size):

        h,w=size

        # coordinate space grid
        x=torch.linspace(-1,1,h).view(1,1,-1)
        x=x.repeat(batch_size,h,1)
        y=torch.linspace(-1,1,w).view(1,-1,1)
        y=y.repeat(batch_size,1,w)
        x,y=Variable(x.cuda(), requires_grad=True),Variable(y.cuda(), requires_grad=True)

        pert=self.get_func(x,y)
        # We find that max and min in \Delta I are usually numerically small
        # Therefore, normalization of \Delta I is not a good choice, because it will enlarge the perturbation
        # So, we omit this step
        # pert=self.norm_pert(pert)
        return pert

    '''Assistant operations'''

    def norm_pert(self,pert):
        # norm the pert into [0,1]
        max_elem=self.max_elem_in_tensor(pert)
        min_elem=self.min_elem_in_tensor(pert)

        pert=(pert-min_elem)/(max_elem-min_elem)
        return pert

    def max_elem_in_tensor(self,tensor):
        max_in_tensor,_=tensor.max(2)
        max_in_tensor,_=max_in_tensor.max(1)

        return max_in_tensor.view(-1,1,1)

    def min_elem_in_tensor(self,tensor):
        min_in_tensor,_=tensor.min(2)
        min_in_tensor,_=min_in_tensor.min(1)

        return min_in_tensor.view(-1,1,1)

    def rotate(self, x, y, cos):
        sin = torch.sqrt(1 - cos ** 2)
        x_new = cos.view(-1, 1, 1) * (x - self.mu_x) + sin.view(-1, 1, 1) * (y - self.mu_y) + self.mu_x
        y_new = -sin.view(-1, 1, 1) * (x - self.mu_x) + cos.view(-1, 1, 1) * (y - self.mu_y) + self.mu_y

        return x_new, y_new



'''
    Gaussian Mixture Model Network
'''

class GMM_Network(nn.Module):
    def __init__(self, target_model, batch_size, mu_x, mu_y, p, pert_scale, Gaussian_number):
        super(GMM_Network, self).__init__()

        self.target_model = target_model

        self.number = Gaussian_number
        self.GMM_layers = nn.Sequential(OrderedDict([(str(i), GMM(target_model, batch_size, 1, 1, mu_x[i], mu_y[i], p[i], pert_scale)) for i in range(self.number)]))

        self.mean=torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1).cuda()
        self.std=torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1).cuda()

    def forward(self, x):

        for i in range(self.number):
            if i == 0:
                pert, out = self.GMM_layers._modules[str(i)](x)
            else:
                pert_new, out = self.GMM_layers._modules[str(i)](out)
                pert = pert + pert_new

        out_norm = self.norm_img(out)
        pred = self.target_model(out_norm)

        return pred, pert, out, out_norm

    def norm_img(self,tensor):
        new_tensor =(tensor-self.mean)/self.std
        return new_tensor

