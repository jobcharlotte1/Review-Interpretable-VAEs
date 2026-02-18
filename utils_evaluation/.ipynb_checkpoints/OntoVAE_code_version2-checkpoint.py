import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from typing import Iterable
import sys
sys.path.append('/home/BS94_SUR/phD/review/models reproductibility/OntoVAE/cobra-ai')
from cobra_ai.module.ontobj import *
from cobra_ai.module.utils import *
from cobra_ai.model.onto_vae import *
from cobra_ai.model.cobra import *
from cobra_ai.module.autotune import *


class Encoder2(nn.Module):
    """
    This class constructs an Encoder module for a variational autoencoder.
    Inspired by SCVI FCLayers class.

    Parameters
    ----------
    in_features
        # of features that are used as input
    latent_dim 
        latent dimension
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    hidden_layers
        number of hidden layers
    neurons_per_layer
        number of neurons per hidden layer
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    activation_fn
        Which activation function to use
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer (True), or just the first (False).
    drop
        dropout rate
    """

    def __init__(self, 
                 device:str,
                 in_features: int, 
                 latent_dim: int, 
                 n_cat_list: Iterable[int] = None,
                 hidden_layers: int = 1,
                 neurons_per_layer: int = 512,
                 use_batch_norm: bool = True,
                 use_layer_norm: bool = False,
                 use_activation: bool = True,
                 activation_fn: nn.Module = nn.ReLU,
                 bias: bool = True,
                 inject_covariates: bool = True,
                 drop: float = 0.2):
        super().__init__()

        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.in_features = in_features
        self.layer_dims = [neurons_per_layer] * hidden_layers
        self.layer_nums = [self.layer_dims[i:i+2] for i in range(len(self.layer_dims)-1)]
        self.latent_dim = latent_dim
        self.drop = drop

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.inject_covariates = inject_covariates
        self.cat_dim = sum(self.n_cat_list)

        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.in_features + self.cat_dim, self.layer_dims[0], bias=bias),
                    nn.BatchNorm1d(self.layer_dims[0]) if use_batch_norm else None,
                    nn.LayerNorm(self.layer_dims[0]) if use_layer_norm else None,
                    activation_fn() if use_activation else None,
                    nn.Dropout(p=self.drop) if self.drop > 0 else None
                )
            ] +

            [build_block(ins = x[0],
                outs = x[1],
                cat_dim = self.cat_dim,
                use_batch_norm = use_batch_norm,
                use_layer_norm = use_layer_norm,
                use_activation = use_activation,
                activation_fn = activation_fn,
                bias = bias,
                inject_covariates = inject_covariates,
                drop = self.drop
            ) for x in self.layer_nums] 
        ).to(self.device)

        self.mu = nn.Sequential(
            nn.Linear(self.layer_dims[-1] + self.cat_dim * inject_covariates, self.latent_dim),
        ).to(self.device)

        self.logvar = nn.Sequential(
            nn.Linear(self.layer_dims[-1] + self.cat_dim * inject_covariates, self.latent_dim),
        ).to(self.device)


    def forward(self, x: torch.tensor, cat_list: Iterable[torch.tensor]):
        """
        Forward computation on minibatch of samples.
        
        Parameters
        ----------
        x
            torch.tensor of shape (minibatch, in_features)
        cat_list
            Iterable of torch.tensors containing the category memberships
            shape of each tensor is (minibatch, 1)
        """

        if self.cat_dim > 0:
            categs = []
            for n_cat, cat in zip(self.n_cat_list, cat_list):
                if n_cat > 1:
                    categs.append(one_hot(cat.long(), n_cat).squeeze())
            categs = torch.hstack(categs)
            c = torch.hstack((x, categs))
        else:
            c = x

        for i, block in enumerate(self.encoder):
            if i == 0:
                for layer in block:
                    if layer is not None:
                        c = layer(c)
            else:
                for layer in block:
                    if layer is not None:
                        if self.cat_dim > 0 and self.inject_covariates and isinstance(layer, nn.Linear):
                            c = layer(torch.hstack((c, categs)))
                        else:
                            c = layer(c)
    
        if self.cat_dim > 0 and self.inject_covariates :
            c = torch.hstack((c, categs))

        mu = self.mu(c)
        log_var = self.logvar(c)

        return mu, log_var



class Decoder2(nn.Module):
    """
    This class constructs a Decoder module for a variational autoencoder.
    Inspired by SCVI FCLayers class.

    Parameters
    ----------
    in_features
        # of features that will be reconstructed
    latent_dim
        input dimension
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    hidden_layers
        number of hidden layers
    neurons_per_layer
        number of neurons per hidden layer
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    activation_fn
        Which activation function to use
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer (True), or just the first (False).
    drop
        dropout rate
    """

    def __init__(self, 
                 device:str,
                 in_features: int, 
                 latent_dim: int, 
                 n_cat_list: Iterable[int] = None,
                 hidden_layers: int = 1,
                 neurons_per_layer: int = 512,
                 use_batch_norm: bool = True,
                 use_layer_norm: bool = False,
                 use_activation: bool = True,
                 activation_fn: nn.Module = nn.ReLU,
                 bias: bool = True,
                 inject_covariates: bool = True,
                 drop: float = 0.2):
        super().__init__()

        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.in_features = in_features
        self.latent_dim = latent_dim
        self.layer_dims = [neurons_per_layer] * hidden_layers
        self.layer_nums = [self.layer_dims[i:i+2] for i in range(len(self.layer_dims)-1)]
        self.drop = drop

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.inject_covariates = inject_covariates
        self.cat_dim = sum(self.n_cat_list)

        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.latent_dim + self.cat_dim, self.layer_dims[0], bias=bias),
                    nn.BatchNorm1d(self.layer_dims[0]) if use_batch_norm else None,
                    nn.LayerNorm(self.layer_dims[0]) if use_layer_norm else None,
                    activation_fn() if use_activation else None,
                    nn.Dropout(p=self.drop) if self.drop > 0 else None
                )
            ] +

            [build_block(ins = x[0],
                outs = x[1],
                cat_dim = self.cat_dim,
                use_batch_norm = use_batch_norm,
                use_layer_norm = use_layer_norm,
                use_activation = use_activation,
                activation_fn = activation_fn,
                bias = bias,
                inject_covariates = inject_covariates,
                drop = self.drop
            ) for x in self.layer_nums] +

            [
                nn.Sequential(
                    nn.Linear(self.layer_dims[-1] + self.cat_dim * self.inject_covariates, self.in_features)
                )
            ]
            ).to(self.device)
       

    def forward(self, x: torch.tensor, cat_list: Iterable[torch.tensor]):
        """
        Forward computation on minibatch of samples.
        
        Parameters
        ----------
        x
            torch.tensor of shape (minibatch, in_features)
        cat_list
            Iterable of torch.tensors containing the category memberships
            shape of each tensor is (minibatch, 1)
        """

        if self.cat_dim > 0:
            categs = []
            for n_cat, cat in zip(self.n_cat_list, cat_list):
                if n_cat > 1:
                    categs.append(one_hot(cat.long(), n_cat).squeeze())
            categs = torch.hstack(categs)
            c = torch.hstack((x, categs))
        else:
            c = x

        for i, block in enumerate(self.decoder[:-1]):
            if i == 0:
                for layer in block:
                    if layer is not None:
                        c = layer(c)
            else:
                for layer in block:
                    if layer is not None:
                        if self.cat_dim > 0 and self.inject_covariates and isinstance(layer, nn.Linear):
                            c = layer(torch.hstack((c, categs)))
                        else:
                            c = layer(c)
    
        if self.cat_dim > 0 and self.inject_covariates :
            out = torch.hstack((c, categs))
        else:
            out = c

        for layer in self.decoder[-1]:
            if layer is not None:
                out = layer(out)
        
        return out


class OntoDecoder2(nn.Module):
    """
    This class constructs an ontology structured Decoder module.
  
    Parameters
    ----------
    in_features
        # of features that are used as input
    layer_dims
        list of tuples that define in and out for each layer
    mask_list
        matrix for each layer transition, that determines which weights to zero out
    root_layer_latent
        whether latent space layer is set as first ontology layer (True, default) or first decoder layer (False)
    latent_dim
        latent dimension
    neuronnum
        number of neurons to use per term
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    activation_fn
        Which activation function to use
    rec_activation
        activation function for the reconstruction layer, eg. nn.Sigmoid
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer (True), or just the last (False).
    drop
        dropout rate
    pos_weights
        whether to make all decoder weights positive
    """ 

    def __init__(self, 
                 device: str,
                 in_features: int, 
                 layer_dims: list, 
                 mask_list: list, 
                 root_layer_latent: bool = True,
                 latent_dim: int = 128, 
                 neuronnum: int = 3,
                 n_cat_list: Iterable[int] = None,
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 use_activation: bool = False,
                 activation_fn: nn.Module = nn.ReLU,
                 rec_activation: nn.Module = None,
                 bias: bool = True,
                 inject_covariates: bool = False,
                 drop: float = 0,
                 pos_weights: bool = True):
        super().__init__()

        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.in_features = in_features
        self.root_layer_latent = root_layer_latent
        self.start_point = 0 if root_layer_latent else 1
        self.layer_dims = np.hstack([layer_dims[:-1] * neuronnum, layer_dims[-1]])
        self.layer_shapes = [(np.sum(self.layer_dims[:i+1]), self.layer_dims[i+1]) for i in range(len(self.layer_dims)-1)]
        self.masks = []
        for m in mask_list[0:-1]:
            m = m.repeat_interleave(neuronnum, dim=0)
            m = m.repeat_interleave(neuronnum, dim=1)
            self.masks.append(m.to(self.device))
        self.masks.append(mask_list[-1].repeat_interleave(neuronnum, dim=1).to(self.device))
        self.latent_dim = latent_dim
        self.drop = drop
        self.pos_weights = pos_weights

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.inject_covariates = inject_covariates
        self.cat_dim = sum(self.n_cat_list)

        self.decoder = nn.ModuleList(

            [build_block(ins = x[0],
                outs = x[1],
                cat_dim = self.cat_dim,
                use_batch_norm = use_batch_norm,
                use_layer_norm = use_layer_norm,
                use_activation = use_activation,
                activation_fn = activation_fn,
                bias = bias,
                inject_covariates = inject_covariates,
                drop = self.drop
            ) for x in self.layer_shapes[:-1]] +

            [
                nn.Sequential(
                    nn.Linear(self.layer_shapes[-1][0] + self.cat_dim, self.in_features),
                    rec_activation() if rec_activation is not None else None
                )
            ]
            )
        
        if not root_layer_latent:
            self.decoder.insert(0, 
                build_block(ins = self.latent_dim,
                            outs = self.layer_dims[0],
                            cat_dim = self.cat_dim,
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            use_activation = use_activation,
                            activation_fn = activation_fn,
                            bias = bias,
                            inject_covariates = inject_covariates,
                            drop = self.drop
                            )
            )

        self.decoder.to(self.device)
        
        # attach covs to masks (set to 1s)
        if len(self.n_cat_list) > 0:
            if inject_covariates:
                self.layer_shapes = [(lshape[0] + self.cat_dim, lshape[1]) for lshape in self.layer_shapes]
                self.masks = [torch.hstack((mask, torch.ones(mask.shape[0], self.cat_dim).to(self.device))) for mask in self.masks]
            else:
                self.layer_shapes[-1] = (self.layer_shapes[-1][0] + self.cat_dim, self.layer_shapes[-1][1]) 
                self.masks[-1] = torch.hstack((self.masks[-1], torch.ones(self.masks[-1].shape[0], self.cat_dim).to(self.device))) 

        # apply masks to zero out weights of non-existent connections
        for i in range(self.start_point,len(self.decoder)):
            self.decoder[i][0].weight.data = torch.mul(self.decoder[i][0].weight.data, self.masks[i-self.start_point])

        # make all weights in decoder positive
        if self.pos_weights:
            for i in range(self.start_point, len(self.decoder)):
                self.decoder[i][0].weight.data = self.decoder[i][0].weight.data.clamp(0)


    def forward(self, z: torch.tensor, cat_list: Iterable[torch.tensor]):
        """
        Forward computation on minibatch of samples.
        
        Parameters
        ----------
        z
            torch.tensor of shape (minibatch, in_features)
        cat_list
            Iterable of torch.tensors containing the category memerships
            shape of each tensor is (minibatch, 1)
        """

        if self.cat_dim > 0:
            categs = []
            for n_cat, cat in zip(self.n_cat_list, cat_list):
                if n_cat > 1:
                    categs.append(one_hot(cat.long(), n_cat).squeeze())
            categs = torch.hstack(categs)

        if not self.root_layer_latent:
            for layer in self.decoder[0]:
                if layer is not None:
                    if self.cat_dim > 0 and self.inject_covariates and isinstance(layer, nn.Linear):
                        z = layer(torch.hstack((z, categs)))
                    else:
                        z = layer(z)
        
        out = z.clone()

        for block in self.decoder[self.start_point:-1]:
            for layer in block:
                if layer is not None:
                    if self.cat_dim > 0 and self.inject_covariates and isinstance(layer, nn.Linear):
                        z = layer(torch.hstack((z, categs)))
                    else:
                        z = layer(z)
            out = torch.cat((z, out), dim=1)
            z = out.clone()
        
        if self.cat_dim > 0:
            out = torch.hstack((out, categs))

        for layer in self.decoder[-1]:
            if layer is not None:
                out = layer(out)
        
        return out
    
    
class Classifier(nn.Module):
    """
    Classifier module that can do binary or multi-class classification
    Parameters
    -------------
    in_features
        # of features that are used as input
    n_classes 
        number of classes
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    hidden_layers
        number of hidden layers
    neurons_per_layer
        number of neurons in a hidden layer
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    activation_fn
        Which activation function to use
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer (True), or just the first (False).
    drop
        dropout rate
    """
    def __init__(self, 
                 in_features: int, 
                 n_classes: int, 
                 n_cat_list: Iterable[int] = None,
                 hidden_layers: int = 1,
                 neurons_per_layer: int = 64,
                 use_batch_norm: bool = True,
                 use_layer_norm: bool = False,
                 use_activation: bool = True,
                 activation_fn: nn.Module = nn.ReLU,
                 bias: bool = True,
                 inject_covariates: bool = True,
                 drop: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.layer_dims = [neurons_per_layer] * hidden_layers
        self.layer_nums = [self.layer_dims[i:i+2] for i in range(len(self.layer_dims)-1)]
        self.n_classes = n_classes
        self.drop = drop
        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []
        self.inject_covariates = inject_covariates
        self.cat_dim = sum(self.n_cat_list)

        self.classifier = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.in_features + self.cat_dim, self.layer_dims[0], bias=bias),
                    nn.BatchNorm1d(self.layer_dims[0]) if use_batch_norm else None,
                    nn.LayerNorm(self.layer_dims[0]) if use_layer_norm else None,
                    activation_fn() if use_activation else None,
                    nn.Dropout(p=self.drop) if self.drop > 0 else None
                )
            ] +
            [build_block(ins = x[0],
                outs = x[1],
                cat_dim = self.cat_dim,
                use_batch_norm = use_batch_norm,
                use_layer_norm = use_layer_norm,
                use_activation = use_activation,
                activation_fn = activation_fn,
                bias = bias,
                inject_covariates = inject_covariates,
                drop = self.drop
            ) for x in self.layer_nums] +
            [
                nn.Sequential(
                    nn.Linear(self.layer_dims[-1] + self.cat_dim * inject_covariates, self.n_classes, bias=bias),
                    nn.BatchNorm1d(self.n_classes) if use_batch_norm else None,
                    nn.LayerNorm(self.n_classes) if use_layer_norm else None,
                    nn.Softmax(dim=1) if self.n_classes > 2 else nn.Sigmoid()
                )
            ]

        )
    
    def forward(self, x: torch.tensor, cat_list: Iterable[torch.tensor]):
        """
        Forward computation on minibatch of samples.
        
        Parameters
        ----------
        x
            torch.tensor of shape (minibatch, in_features)
        cat_list
            Iterable of torch.tensors containing the category memberships
            shape of each tensor is (minibatch, 1)
        """
        if self.cat_dim > 0:
            categs = []
            for n_cat, cat in zip(self.n_cat_list, cat_list):
                if n_cat > 1:
                    categs.append(one_hot(cat.long(), n_cat).squeeze())
            categs = torch.hstack(categs)
            c = torch.hstack((x, categs))
        else:
            c = x
        for i, block in enumerate(self.classifier):
            if i == 0:
                for layer in block:
                    if layer is not None:
                        c = layer(c)
            else:
                for layer in block:
                    if layer is not None:
                        if self.cat_dim > 0 and self.inject_covariates and isinstance(layer, nn.Linear):
                            c = layer(torch.hstack((c, categs)))
                        else:
                            c = layer(c)

        return c
    
"""Function to build NN blocks"""

def build_block(ins: int,
                outs: int,
                cat_dim: int,
                use_batch_norm: bool = True,
                use_layer_norm: bool = False,
                use_activation: bool = True,
                activation_fn: nn.Module = nn.ReLU,
                bias: bool = True,
                inject_covariates: bool = True,
                drop: float = 0.2, 
                ):
    return nn.Sequential(
            nn.Linear(ins + cat_dim * inject_covariates, outs, bias=bias),
            nn.BatchNorm1d(outs) if use_batch_norm else None,
            nn.LayerNorm(outs) if use_layer_norm else None,
            activation_fn() if use_activation else None,
            nn.Dropout(p=drop) if drop > 0 else None
    )
  



class OntoVAE2(nn.Module):
    """
    This class combines a normal encoder with an ontology structured decoder.
    The input should be log-transformed normalized data. 
    Mainly for single-cell, but also works with bulk data if stored in adata.

    Parameters
    ----------
    adata
        anndata object that has been preprocessed with setup_anndata_ontovae function
    use_batch_norm_enc
        Whether to have `BatchNorm` layers or not in encoder
    use_layer_norm_enc
        Whether to have `LayerNorm` layers or not in encoder
    use_activation_enc
        Whether to have layer activation or not in encoder
    activation_fn_enc
        Which activation function to use in encoder
    bias_enc
        Whether to learn bias in linear layers or not in encoder
    hidden_layers_enc
        number of hidden layers in encoder (number of nodes is determined by neuronnum)
    inject_covariates_enc
        Whether to inject covariates in each layer (True), or just the first (False) of encoder
    drop_enc
        dropout rate in encoder
    z_drop
        dropout rate for latent space 
    root_layer_latent
        whether latent space layer is set as first ontology layer (True, default) or first decoder layer (False)
    latent_dim
        latent space dimension if root_layer_latent is False
    neuronnum
        number of neurons per term in decoder
    use_batch_norm_dec
        Whether to have `BatchNorm` layers or not in decoder
    use_layer_norm_dec
        Whether to have `LayerNorm` layers or not in decoder
    use_activation_dec
        Whether to have layer activation or not in decoder
    use_activation_lat
        Whether to use the decoder activation function after latent space sampling (not recommended)
    activation_fn_dec
        Which activation function to use in decoder
    rec_activation
        activation function for the reconstruction layer, e.g. nn.Sigmoid
    bias_dec
        Whether to learn bias in linear layers or not in decoder
    inject_covariates_dec
        Whether to inject covariates in each layer (True), or just the last (False) of decoder
    drop_dec
        dropout rate in decoder
    """

    @classmethod
    def load(cls, adata: AnnData, modelpath: str, device:str):
        with open(modelpath + '/model_params.json', 'r') as fp:
            params = json.load(fp)
            
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Add device to params
        params['device'] = device
            
        if params['activation_fn_enc'] is not None:
            params['activation_fn_enc'] = eval(params['activation_fn_enc'])
        if params['activation_fn_dec'] is not None:
            params['activation_fn_dec'] = eval(params['activation_fn_dec'])
        if params['rec_activation'] is not None:
            params['rec_activation'] = eval(params['rec_activation'])
        model = cls(adata, **params) 
        checkpoint = torch.load(modelpath + '/best_model.pt',
                            map_location = torch.device(model.device))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        return model

    def __init__(self, 
                 adata: AnnData, 
                 device: str,
                 use_batch_norm_enc: Tunable[bool] = True,
                 use_layer_norm_enc: Tunable[bool] = False,
                 use_activation_enc: Tunable[bool] = True,
                 activation_fn_enc: Tunable[nn.Module] = nn.ReLU,
                 bias_enc: Tunable[bool] = True,
                 hidden_layers_enc: Tunable[int]=2, 
                 inject_covariates_enc: Tunable[bool] = False,
                 drop_enc: Tunable[float] = 0.2, 
                 z_drop: Tunable[float] = 0.5,
                 root_layer_latent: Tunable[bool] = False,
                 latent_dim: Tunable[int] = 128,
                 neuronnum: Tunable[int] = 3,
                 use_batch_norm_dec: Tunable[bool] = False,
                 use_layer_norm_dec: Tunable[bool] = False,
                 use_activation_dec: Tunable[bool] = False,
                 use_activation_lat: Tunable[bool] = False,
                 activation_fn_dec: Tunable[nn.Module] = nn.ReLU,
                 rec_activation: nn.Module = None,
                 bias_dec: Tunable[bool] = True,
                 inject_covariates_dec: Tunable[bool] = False,
                 drop_dec: Tunable[float] = 0):
        super().__init__()
        
        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # move model to device
        self.to(self.device)

        # store init params in dict
        self.params = {'use_batch_norm_enc': use_batch_norm_enc,
                          'use_layer_norm_enc': use_layer_norm_enc,
                          'use_activation_enc': use_activation_enc,
                          'activation_fn_enc': str(activation_fn_enc).split("'")[1] if activation_fn_enc is not None else activation_fn_enc,
                          'bias_enc': bias_enc,
                          'hidden_layers_enc': hidden_layers_enc,
                          'inject_covariates_enc': inject_covariates_enc,
                          'drop_enc': drop_enc,
                          'z_drop': z_drop,
                          'root_layer_latent': root_layer_latent,
                          'latent_dim': latent_dim,
                          'neuronnum': neuronnum,
                          'use_batch_norm_dec': use_batch_norm_dec,
                          'use_layer_norm_dec': use_layer_norm_dec,
                          'use_activation_dec': use_activation_dec,
                          'use_activation_lat': use_activation_lat,
                          'activation_fn_dec': str(activation_fn_dec).split("'")[1] if activation_fn_dec is not None else activation_fn_dec,
                          'rec_activation': str(rec_activation).split("'")[1] if rec_activation is not None else rec_activation,
                          'bias_dec': bias_dec,
                          'inject_covariates_dec': inject_covariates_dec,
                          'drop_dec': drop_dec}


        self.adata = adata

        if '_ontovae' not in self.adata.uns.keys():
            raise ValueError('Please run cobra_ai.module.utils.setup_anndata_ontovae first.')

        # parse OntoVAE information
        self.thresholds = adata.uns['_ontovae']['thresholds']
        self.in_features = len(self.adata.uns['_ontovae']['genes'])
        self.mask_list = adata.uns['_ontovae']['masks']
        self.mask_list = [torch.tensor(m, dtype=torch.float32) for m in self.mask_list]
        self.layer_dims_dec =  np.array([self.mask_list[0].shape[1]] + [m.shape[0] for m in self.mask_list])
        self.root_layer_latent = root_layer_latent
        self.start_point = 0 if self.root_layer_latent else 1
        self.latent_dim = self.layer_dims_dec[0] * neuronnum if self.root_layer_latent else latent_dim
        self.neurons_per_layer_enc = self.latent_dim
        self.z_drop = z_drop

        # additional info
        self.neuronnum = neuronnum
        self.use_activation_dec = use_activation_dec
        self.use_activation_lat = use_activation_lat
        self.activation_fn_dec = activation_fn_dec
        self.rec_activation = rec_activation

        # parse covariate information
        self.batch = adata.obs['_ontovae_batch']
        self.labels = adata.obs['_ontovae_labels']
        self.covs = adata.obsm['_ontovae_categorical_covs'] if '_ontovae_categorical_covs' in adata.obsm.keys() else None

        self.n_cat_list = [len(self.batch.unique()), len(self.labels.unique())]
        if self.covs is not None:
            self.n_cat_list.extend([len(self.covs[c].unique()) for c in self.covs.columns])

        self.rec_weights = None

        # Encoder
        self.encoder = Encoder2(device = self.device,
                                in_features = self.in_features,
                                latent_dim = self.latent_dim,
                                n_cat_list = self.n_cat_list,
                                hidden_layers = hidden_layers_enc,
                                neurons_per_layer = self.neurons_per_layer_enc, 
                                use_batch_norm = use_batch_norm_enc,
                                use_layer_norm = use_layer_norm_enc,
                                use_activation = use_activation_enc,
                                activation_fn = activation_fn_enc,
                                bias = bias_enc,
                                inject_covariates = inject_covariates_enc,
                                drop = drop_enc)

        # Decoder
        self.decoder = OntoDecoder2(device=self.device,
                                    in_features = self.in_features,
                                    layer_dims = self.layer_dims_dec,
                                    mask_list = self.mask_list,
                                    root_layer_latent = self.root_layer_latent,
                                    latent_dim = self.latent_dim,
                                    neuronnum = self.neuronnum,
                                    n_cat_list = self.n_cat_list,
                                    use_batch_norm = use_batch_norm_dec,
                                    use_layer_norm = use_layer_norm_dec,
                                    use_activation = use_activation_dec,
                                    activation_fn = activation_fn_dec,
                                    rec_activation = rec_activation,
                                    bias = bias_dec,
                                    inject_covariates = inject_covariates_dec,
                                    drop = drop_dec)

        self.to(self.device)

    def _cov_tensor(self, adata):
        """
        Helper function to aggregate information from adata to use as input for dataloader.
        """
        covs = adata.obs[['_ontovae_batch', '_ontovae_labels']]
        if '_ontovae_categorical_covs' in adata.obsm.keys():
            covs = pd.concat([covs, adata.obsm['_ontovae_categorical_covs']], axis=1)
        return torch.tensor(np.array(covs))

    def reparameterize(self, mu, log_var):
        """
        Performs the reparameterization trick.

        Parameters
        ----------
        mu
            mean from the encoder's latent space
        log_var
            log variance from the encoder's latent space
        mode
            train: training mode
            val: validation mode
        """
        sigma = torch.exp(0.5*log_var) 
        eps = torch.randn_like(sigma) 
        z = mu + eps * sigma
        if self.z_drop > 0:
            if self.training:
                z = nn.Dropout(p=self.z_drop)(z)
        return z
        
    def _get_embedding(self, x: torch.tensor, cat_list: Iterable[torch.tensor]):
        """
        Generates latent space embedding.

        Parameters
        ----------
        x
            torch.tensor of shape (minibatch, in_features)
        cat_list
            Iterable of torch.tensors containing the category memberships
            shape of each tensor is (minibatch, 1)
        """
        mu, log_var = self.encoder(x, cat_list)
        z = self.reparameterize(mu, log_var)
        if self.use_activation_lat:
            z = self.activation_fn_dec()(z)
        return z, mu, log_var


    def forward(self, x: torch.tensor, cat_list: Iterable[torch.tensor]):
        """
        Forward computation on minibatch of samples.
        
        Parameters
        ----------
        x
            torch.tensor of shape (minibatch, in_features)
        cat_list
            Iterable of torch.tensors containing the category memberships
            shape of each tensor is (minibatch, 1)
        """

        z, mu, log_var = self._get_embedding(x, cat_list)
        reconstruction = self.decoder(z, cat_list)
        return z, mu, log_var, reconstruction

    def vae_loss(self, reconstruction, mu, log_var, data, kl_coeff, run=None):
        """
        Calculates VAE loss as combination of reconstruction loss and weighted Kullback-Leibler loss.
        """
        kl_loss = -0.5 * torch.sum(1. + log_var - mu.pow(2) - log_var.exp(), )
        if self.rec_weights is not None:
            rec_loss = torch.sum(torch.matmul(input=(data - reconstruction)**2, other=self.rec_weights))
        else:
            rec_loss = F.mse_loss(reconstruction, data, reduction="sum")
        if run is not None:
            mode = 'train' if self.training else 'val'
            run["metrics/" + mode + "/kl_loss"].log(kl_loss)
            run["metrics/" + mode + "/rec_loss"].log(rec_loss)
        return torch.mean(rec_loss + kl_coeff*kl_loss)
    
    def train_round(self, 
                    dataloader: FastTensorDataLoader, 
                    kl_coeff: float, 
                    optimizer: optim.Optimizer, 
                    pos_weights: bool,
                    run=None):
        """
        Parameters
        ----------
        dataloader
            pytorch dataloader instance with training data
        kl_coeff 
            coefficient for weighting Kullback-Leibler loss
        optimizer
            optimizer for training
        run
            Neptune run if training is to be logged
        """
        # set to train mode
        self.train()

        # initialize running loss
        running_loss = 0.0

        # iterate over dataloader for training
        for i, minibatch in tqdm(enumerate(dataloader), total=len(dataloader)):

            # move minibatch to device
            data = minibatch[0].to(self.device)
            cat_list = torch.split(minibatch[1].T.to(self.device), 1)
            optimizer.zero_grad()

            # forward step
            _, mu, log_var, reconstruction = self.forward(data, cat_list)
            loss = self.vae_loss(reconstruction, mu, log_var, data, kl_coeff, run=run)
            running_loss += loss.item()

            # backward propagation
            loss.backward()

            # zero out gradients from non-existent connections
            for i in range(self.start_point, len(self.decoder.decoder)):
                self.decoder.decoder[i][0].weight.grad = torch.mul(self.decoder.decoder[i][0].weight.grad, self.decoder.masks[i-self.start_point])

            # perform optimizer step
            optimizer.step()

            # make weights in Onto module positive
            if pos_weights:
                for i in range(self.start_point, len(self.decoder.decoder)):
                    self.decoder.decoder[i][0].weight.data = self.decoder.decoder[i][0].weight.data.clamp(0)

        # compute avg training loss
        train_loss = running_loss/len(dataloader)
        return train_loss

    @torch.no_grad()
    def val_round(self, 
                  dataloader: FastTensorDataLoader, 
                  kl_coeff: float, 
                  run=None):
        """
        Parameters
        ----------
        dataloader
            pytorch dataloader instance with training data
        kl_coeff
            coefficient for weighting Kullback-Leibler loss
        run
            Neptune run if training is to be logged
        """
        # set to eval mode
        self.eval()

        # initialize running loss
        running_loss = 0.0

        # iterate over dataloader for validation
        for i, minibatch in tqdm(enumerate(dataloader), total=len(dataloader)):

            # move minibatch to device
            data = minibatch[0].to(self.device)
            cat_list = torch.split(minibatch[1].T.to(self.device), 1)

            # forward step
            _, mu, log_var, reconstruction = self.forward(data, cat_list)
            loss = self.vae_loss(reconstruction, mu, log_var,data, kl_coeff, run=run)
            running_loss += loss.item()

        # compute avg val loss
        val_loss = running_loss/len(dataloader)
        return val_loss

    def train_model(self, 
                    modelpath: str, 
                    save: bool = True,
                    train_size: float = 0.9,
                    seed: int = 42,
                    lr: Tunable[float]=1e-4, 
                    kl_coeff: Tunable[float]=1e-4, 
                    batch_size: Tunable[int]=128, 
                    optimizer: Tunable[optim.Optimizer] = optim.AdamW,
                    pos_weights: Tunable[bool] = True,
                    use_rec_weights: bool = False,
                    epochs: int=300, 
                    early_stopping: bool=True,
                    patience: int=10,
                    run=None):
        """
        Parameters
        ----------
        modelpath
            path to a folder where to store the params and the best model 
        save
            if the params and the best model should be saved, if save is False the modelpath parameter could only be an empty string
        train_size
            which percentage of samples to use for training
        seed
            seed for the train-val split
        lr
            learning rate
        kl_coeff
            Kullback Leibler loss coefficient
        batch_size
            size of minibatches
        optimizer
            which optimizer to use
        pos_weights
            whether to make weights in decoder positive
        epochs
            over how many epochs to train
        run
            passed here if logging to Neptune should be carried out
        """

        if os.path.isfile(modelpath + '/best_model.pt'):
            print("A model already exists in the specified directory and will be overwritten.")

        if save:
            # save train params
            train_params = {'train_size': train_size,
                            'seed': seed,
                            'lr': lr,
                            'kl_coeff': kl_coeff,
                            'batch_size': batch_size,
                            'optimizer': str(optimizer).split("'")[1],
                            'pos_weights': pos_weights,
                            'use_rec_weights': use_rec_weights,
                            'epochs': epochs,
                            'early_stopping': early_stopping,
                            'patience': patience
                            }
            with open(modelpath + '/train_params.json', 'w') as fp:
                json.dump(train_params, fp, indent=4)
            
            if run is not None:
                run["train_parameters"] = train_params

            # save model params
            with open(modelpath + '/model_params.json', 'w') as fp:
                json.dump(self.params, fp, indent=4)
            
            if run is not None:
                run["model_parameters"] = self.params

        # train-val split
        train_adata, val_adata = split_adata(self.adata, 
                                             train_size = train_size,
                                             seed = seed)

        train_covs = self._cov_tensor(train_adata)
        val_covs = self._cov_tensor(val_adata)

        # generate dataloaders
        trainloader = FastTensorDataLoader(torch.tensor(train_adata.X.todense(), dtype=torch.float32), 
                                           train_covs,
                                         batch_size=batch_size, 
                                         shuffle=True)
        valloader = FastTensorDataLoader(torch.tensor(val_adata.X.todense(), dtype=torch.float32), 
                                         val_covs,
                                        batch_size=batch_size, 
                                        shuffle=False)
        
        # compute reconstruction weights
        if use_rec_weights:
            weights = torch.tensor(np.var(np.array(self.adata.X.todense()), axis=0), dtype=torch.float32)
            self.rec_weights = torch.mul(weights, torch.div(weights[weights != 0].size(dim=0), torch.sum(weights,))).to(self.device)
        else:
            self.rec_weights = None

        val_loss_min = float('inf')
        optimizer = optimizer(self.parameters(), lr = lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)

        if early_stopping:
                early_stopper = EarlyStopper(patience=patience)

        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")
            train_epoch_loss = self.train_round(trainloader, kl_coeff, optimizer, pos_weights, run)
            
            scheduler.step()

            val_epoch_loss = self.val_round(valloader, kl_coeff, run)

            if early_stopping:
                if early_stopper.early_stop(val_epoch_loss):
                    break

            #train.report({"validation_loss": val_epoch_loss})

            if run is not None:
                run["metrics/train/loss"].log(train_epoch_loss)
                run["metrics/val/loss"].log(val_epoch_loss)
                
            if val_epoch_loss < val_loss_min and save:
                print('New best model!')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_epoch_loss,
                }, modelpath + '/best_model.pt')
                val_loss_min = val_epoch_loss
                
            print(f"Train Loss: {train_epoch_loss:.4f}")
            print(f"Val Loss: {val_epoch_loss:.4f}")

    def _get_activation(self, index, activation={}):
        def hook(model, input, output):
            activation[index] = output
        return hook 
    
    def _attach_hooks(self, lin_layer=True, activation={}, hooks={}):
        """helper function to attach hooks to the decoder"""
        for i in range(len(self.decoder.decoder)-1):
            key = str(i)
            hook_ind=0 if lin_layer else np.where(np.array(self.decoder.decoder[i]) != None)[0][-1]
            value = self.decoder.decoder[i][hook_ind].register_forward_hook(self._get_activation(i, activation))
            hooks[key] = value


    @torch.no_grad()
    def _hook_activities(self, x, cat_list, lin_layer=True):
        """
        Attaches hooks and retrieves pathway activities.

        Parameters
        ----------
        x
            torch.tensor of shape (minibatch, in_features)
        cat_list
            Iterable of torch.tensors containing the category memberships
            shape of each tensor is (minibatch, 1)
        lin_layer:
            whether hooks should be attached to linear layer of the model
        """

        # set to eval mode
        self.eval()

        # initialize activations and hooks
        activation = {}
        hooks = {}

        # attach the hooks
        self._attach_hooks(lin_layer=lin_layer, activation=activation, hooks=hooks)

        # pass data through model
        z, _, _, _ = self.forward(x, cat_list)

        act = torch.cat(list(activation.values()), dim=1)
        
        # remove hooks
        for h in hooks:
            hooks[h].remove()

        # return pathway activities or reconstructed gene values
        if self.root_layer_latent:
            return torch.hstack((z,act))
        else:
            return act

    def _average_neuronnum(self, act: np.array):
        """
        Helper function to calculate the average value of multiple neurons.
        """
        act = np.array(np.split(act, act.shape[1]/self.neuronnum, axis=1)).mean(axis=2).T
        return act

    @torch.no_grad()
    def _run_batches(self, adata: AnnData, retrieve: Literal['latent', 'act', 'rec'], lin_layer: bool=True):
        """
        Runs batches of a dataloader through encoder or complete VAE and collects results.

        Parameters
        ----------
        latent
            whether to retrieve latent space embedding (True) or reconstructed values (False)
        """
        self.eval()

        if adata is not None:
            if '_ontovae' not in adata.uns.keys():
                raise ValueError('Please run cobra_ai.module.utils.setup_anndata first.')
        else:
            adata = self.adata

        covs = self._cov_tensor(adata)

        dataloader = FastTensorDataLoader(torch.tensor(adata.X.todense(), dtype=torch.float32),
                                          covs,
                                         batch_size=128, 
                                         shuffle=False)

        res = []
        for minibatch in dataloader:
            x = minibatch[0].to(self.device)
            cat_list = torch.split(minibatch[1].T.to(self.device), 1)
            if retrieve == 'latent':
                result, _, _ = self._get_embedding(x, cat_list)
            elif retrieve == 'act':
                result = self._hook_activities(x, cat_list, lin_layer)
            else:
                _, _, _, result = self.forward(x, cat_list)
            result = result.to('cpu').detach().numpy()
            if retrieve == 'latent':
                if self.root_layer_latent:
                    result = self._average_neuronnum(result)
            if retrieve == 'act':
                result = self._average_neuronnum(result)
            res.append(result)
        res = np.vstack(res)

        return res
    
    @torch.no_grad()
    def to_latent(self, adata: AnnData=None):
        """
        Wrapper around _run_batches to retrieve latent space embedding.

        Parameters
        ----------
        adata
            AnnData object that was processed with setup_anndata_vanillavae
        """
        self.eval()
        res = self._run_batches(adata, retrieve='latent')
        return res
    
    @torch.no_grad()
    def get_pathway_activities(self, adata: AnnData=None, lin_layer=True):
        """
        Wrapper around _run_batches to retrieve pathway activities.

        Parameters
        ----------
        adata
            AnnData object that was processed with setup_anndata
        lin_layer
            whether linear layer should be used for calculation
        """
        if len(self.decoder.decoder) == 1:
            raise ValueError('Pathway activities cannot be computed for a one-layer network.')

        self.eval()
        res = self._run_batches(adata, 'act', lin_layer)
        return res

    @torch.no_grad()
    def get_reconstructed_values(self, adata: AnnData=None):
        """
        Wrapper around _run_batches to retrieve output layer.

        Parameters
        ----------
        adata
            AnnData object that was processed with setup_anndata_vanillavae
        """
        self.eval()
        res = self._run_batches(adata, retrieve='rec')
        return res
    
    @torch.no_grad()
    def perturbation(self, adata: AnnData=None, genes: list=[], values: list=[], output=Literal['latent','act','rec'], lin_layer=True):
        """
        Retrieves pathway activities or reconstructed gene values after performing in silico perturbation.

        Parameters
        ----------
        adata
            AnnData object that was processed with setup_anndata
        genes
            a list of genes to perturb
        values
            list with new values, same length as genes
        output
            whether to retrieve latent space ('latent'), pathway activities ('act') or reconstructed gene values ('rec')
        lin_layer
            whether linear layer should be used for pathway activity retrieval
        """
        self.eval()

        if adata is not None:
            if '_ontovae' not in adata.uns.keys():
                raise ValueError('Please run cobra_ai.module.utils.setup_anndata first.')
            pdata = adata.copy()
        else:
            pdata = self.adata.copy()

        # get indices of the genes in list
        gindices = [pdata.uns['_ontovae']['genes'].index(g) for g in genes]

        # replace their values
        for i in range(len(genes)):
            pdata.X[:,gindices[i]] = values[i]

        # run perturbed data through network
        if output == 'latent':
            res = self._run_batches(pdata, 'latent')
        elif output == 'act':
            res = self._run_batches(pdata, 'act', lin_layer)
        else:
            res = self._run_batches(pdata, retrieve='rec')

        return res


    @classproperty
    def _tunables(cls):
        return [cls.__init__, cls.train_model]
    
    @classproperty
    def _metrics(cls):
        ''' Maybe should provide the metric in the manner ["name", "mode"]'''
        return ["validation_loss"]
