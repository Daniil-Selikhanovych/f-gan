import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

class Generator(nn.Module):
    """
    Generator network. If elliptical == False,
        G(z|b) = z + b,
    If elliptical == True,
        G(z,\\xi|b) = \\xi z + b.
    """
    def __init__(self, p, init_bias = 0.0, elliptical=False):
        """
        Args:
            p: number. p is the dimension of samples.
            elliptical: boolean.
        """
        super(Generator, self).__init__()
        self.p = p
        self.bias = nn.Parameter(torch.ones(self.p) * init_bias)
        self.elliptical = elliptical

    def forward(self, z, xi=None):
        if self.elliptical:
            z = xi * z
        x = z + self.bias
        x = x.view(-1, self.p)
        return x
        
class Discriminator(nn.Module):
    """
    Discriminator network.
    """
    def __init__(self, p, hidden_units, activation_1):
        """
        Args:
            p: sample's dimension.
            hidden_units: a list of hidden units for Discriminator, 
                          e.g. d_hidden_units=[10, 5], then the discrimintor has
                          structure p (input) - 10 - 5 - 1 (output).
            activation_1: 'Sigmoid', 'ReLU' or 'LeakyReLU'. The first activation 
                          function after the input layer. Especially when 
                          true_type == 'Cauchy', Sigmoid activation is preferred.
        """
        super(Discriminator, self).__init__()
        self.p = p
        self.arg_1 = {'negative_slope':0.2} if (activation_1 == 'LeakyReLU') else {}
        self.activation_1 = activation_1
        self.layers = len(hidden_units)
        self.hidden_units = hidden_units
        self.feature = self._make_layers()
        self.map_last = nn.Linear(self.hidden_units[-1], 1)
        
    def forward(self, x):
        x = self.feature(x.view(-1,self.p)) 
        d = self.map_last(x).squeeze()
        return x, d

    def _make_layers(self):
        layer_list = []
        for lyr in range(self.layers):
            if lyr == 0:
                layer_list += [
                    ('lyr%d'%(lyr+1), nn.Linear(self.p, self.hidden_units[lyr])),
                    ('act%d'%(lyr+1), getattr(nn, self.activation_1)(**self.arg_1))
                    ]
            else:
                layer_list += [
                    ('lyr%d'%(lyr+1), 
                        nn.Linear(self.hidden_units[lyr-1], self.hidden_units[lyr])),
                    ('act%d'%(lyr+1), nn.LeakyReLU(0.2))]
        return nn.Sequential(OrderedDict(layer_list))
        
class LogisticRegression(nn.Module):
    def __init__(self, p):
        super(LogisticRegression, self).__init__()
        self.p = p
        layer_list = [
                      ('lyr%d'%(1), torch.nn.Linear(self.p, 1))
                     ]
        self.layer_list = nn.Sequential(OrderedDict(layer_list))
            
    def forward(self, x):
        outputs = self.layer_list(x.view(-1,self.p))
        return x, outputs
