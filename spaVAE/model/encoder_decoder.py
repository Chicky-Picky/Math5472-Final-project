import torch
import torch.nn as nn

from torch.nn import ModuleList, Sequential, ReLU, ELU, Sigmoid, Linear, Conv1d, MaxPool1d, BatchNorm1d, LayerNorm, Dropout


def _dense_NN(network_type: str,
              input_dim: int,
              hidden_dims: list,
              activation_type: str = "relu",
              dropout: float = 0.,
              norm_type: str = "batchnorm",
              ):
    
    if activation_type == "relu":
        activation = ReLU
    elif activation_type == "sigmoid":
        activation = Sigmoid
    elif activation_type == "elu":
        activation = ELU
    else:
        raise Exception("Unknown activation function")
        
    if norm_type == "batchnorm":
        norm = BatchNorm1d
    elif norm_type == "layernorm":
        norm = LayerNorm
    else:
        raise Exception("Unknown normalization method")

    layers = ModuleList()
    if network_type == "encoder" and dropout > 0:
        layers.append(Dropout(p=dropout))
    layers.append(Linear(input_dim, hidden_dims[0]))
    layers.append(norm(hidden_dims[0]))
    layers.append(activation())

    for i, dim in enumerate(hidden_dims[1:]):
        layers.append(Linear(hidden_dims[i], dim))
        layers.append(norm(dim))
        layers.append(activation())
        if dropout > 0:
            layers.append(Dropout(p=dropout))

    return layers


# Activation for decoder mean
class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        # Following authors' suggestion to use exponent to derive mean > 0 and clamp it
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DenseEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list,
                 output_dim: int,
                 activation_type: str = "relu",
                 dropout: float = 0.,
                 norm_type: str = "batchnorm",
                 ):
        super(DenseEncoder, self).__init__()

        self.layers = _dense_NN(network_type='encoder',
                                input_dim=input_dim,
                                hidden_dims=hidden_dims,
                                activation_type=activation_type,
                                dropout=dropout,
                                norm_type=norm_type,
                                )
        self.enc_mu = Linear(hidden_dims[-1], output_dim)
        self.enc_var = Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        # Following authors' suggestion to use exponent to derive variance > 0 and clamp it
        var = torch.exp(self.enc_var(x).clamp(-15, 15))
        mu = self.enc_mu(x)

        return mu, var
    

class DenseDecoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list,
                 output_dim: int,
                 activation_type: str = "relu",
                 dropout: float = 0.,
                 norm_type: str = "batchnorm",
                 ):
        super(DenseDecoder, self).__init__()

        self.layers = _dense_NN(network_type='decoder',
                                input_dim=input_dim,
                                hidden_dims=hidden_dims,
                                activation_type=activation_type,
                                dropout=dropout,
                                norm_type=norm_type,
                                )
        
        self.dec_mu = Sequential(Linear(hidden_dims[-1], output_dim), MeanAct())
        self.dec_var = nn.Parameter(torch.randn(output_dim), requires_grad=True)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        # Following authors' suggestion to use exponent to derive variance > 0 and clamp it
        var = (torch.exp(torch.clamp(self.dec_var, -15., 15.))).unsqueeze(0)
        mu = self.dec_mu(x)

        return mu, var
    

def _conv_NN(input_dim: int,
             hidden_dims: list,
             activation_type: str = "relu",
             dropout: float = 0.,
             norm_type: str = "batchnorm",
             ):
    
    if activation_type == "relu":
        activation = ReLU
    elif activation_type == "sigmoid":
        activation = Sigmoid
    elif activation_type == "elu":
        activation = ELU
    else:
        raise Exception("Unknown activation function")
        
    if norm_type == "batchnorm":
        norm = BatchNorm1d
    elif norm_type == "layernorm":
        norm = LayerNorm
    else:
        raise Exception("Unknown normalization method")

    layers = ModuleList()
    if dropout > 0:
        layers.append(Dropout(p=dropout))
    
    # Add convolution-pool block
    layers.append(Conv1d(1, 2, 5))
    layers.append(activation())
    layers.append(MaxPool1d(2, 2))
    l_out = int((input_dim - 6) / 2 + 1)
    
    layers.append(Linear(l_out * 2, hidden_dims[0]))
    layers.append(norm(hidden_dims[0]))
    layers.append(activation())

    for i, dim in enumerate(hidden_dims[1:]):
        layers.append(Linear(hidden_dims[i], dim))
        layers.append(norm(dim))
        layers.append(activation())
        if dropout > 0:
            layers.append(Dropout(p=dropout))

    return layers


class ConvEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list,
                 output_dim: int,
                 activation_type: str = "relu",
                 dropout: float = 0.,
                 norm_type: str = "batchnorm",
                 ):
        super(ConvEncoder, self).__init__()

        self.dropout = dropout
        self.layers = _conv_NN(input_dim=input_dim,
                               hidden_dims=hidden_dims,
                               activation_type=activation_type,
                               dropout=dropout,
                               norm_type=norm_type,
                               )
        self.enc_mu = Linear(hidden_dims[-1], output_dim)
        self.enc_var = Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)

        if self.dropout > 0:
            _layers, layers_ = self.layers[:4], self.layers[4:]
        else:
            _layers, layers_ = self.layers[:3], self.layers[3:]

        for layer in _layers:
            x = layer(x)
        x = torch.flatten(x, 1)
        for layer in layers_:
            x = layer(x)

        # Following authors' suggestion to use exponent to derive variance > 0 and clamp it
        var = torch.exp(self.enc_var(x).clamp(-15, 15))
        mu = self.enc_mu(x)

        return mu, var