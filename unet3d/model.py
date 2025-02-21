#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import torch.nn as nn
import torch
import numpy as np
from typing import Callable, Optional
import scipy
from statsmodels.stats.inter_rater import fleiss_kappa
from tqdm import tqdm

from unet3d.buildingblocks import Encoder, Decoder, DoubleConv, ExtResNetBlock
from unet3d.utils import number_of_features_per_level


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]

class Abstract3DUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        f_maps (int, tuple): if int: number of feature maps in the first conv layer of the encoder (default: 64);
            if tuple: number of feature maps at each level
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        testing (bool): if True (testing mode) the `final_activation` (if present, i.e. `is_segmentation=true`)
            will be applied as the last operation during the forward pass; if False the model is in training mode
            and the `final_activation` (even if present) won't be applied; default: False
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, testing=False,
                 conv_kernel_size=3, pool_kernel_size=2, conv_padding=1, set_last_bias=True, pi=0.01,
                 do_p_encoder = 0, do_p_decoder = 0, **kwargs):
        super(Abstract3DUNet, self).__init__()

        self.testing = testing
        self.do_p_encoder = do_p_encoder
        self.do_p_decoder = do_p_decoder

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num,
                                  apply_pooling=False,  # skip pooling in the firs encoder
                                  basic_module=basic_module,
                                  conv_layer_order=layer_order,
                                  conv_kernel_size=conv_kernel_size,
                                  num_groups=num_groups,
                                  padding=conv_padding)
            else:
                # TODO: adapt for anisotropy in the data, i.e. use proper pooling kernel to make the data isotropic after 1-2 pooling operations
                encoder = Encoder(f_maps[i - 1], out_feature_num,
                                  basic_module=basic_module,
                                  conv_layer_order=layer_order,
                                  conv_kernel_size=conv_kernel_size,
                                  num_groups=num_groups,
                                  pool_kernel_size=pool_kernel_size,
                                  padding=conv_padding)

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            if basic_module == DoubleConv:
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            else:
                in_feature_num = reversed_f_maps[i]

            out_feature_num = reversed_f_maps[i + 1]
            # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
            # currently strides with a constant stride: (2, 2, 2)
            decoder = Decoder(in_feature_num, out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              padding=conv_padding)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # dropout layer
        self.do_encoder = nn.Dropout3d(p=do_p_encoder)
        self.do_decoder = nn.Dropout3d(p=do_p_decoder)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        
        # Initialize bias of last layer
        if set_last_bias:
            self.final_conv.bias.data[1:].fill_(-np.log((1-pi)/pi))
            
        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            # dont dropout the smallest latent space
            if i != len(self.encoders)-1:
                x = self.do_encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)
            

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)
            x = self.do_decoder(x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if self.testing and self.final_activation is not None:
            x = self.final_activation(x)

        return x
    
    def enable_dropout(self, for_encoder = True, for_decoder = True):
        """ Function to enable the dropout layers during test-time """
        if for_encoder:
            self.do_encoder.train()
        if for_decoder:
            self.do_decoder.train()
            
    def disable_dropout(self, for_encoder = True, for_decoder = True):
        """ Function to enable the dropout layers during test-time """
        if for_encoder:
            self.do_encoder.eval()
        if for_decoder:
            self.do_decoder.eval()
            
    def attach_post_processor(self, 
                              post_processor: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                              post_proc_kwargs: Optional[dict] = None):
        self.post_processor = post_processor
        self.post_proc_kwargs = post_proc_kwargs
            
    def MC_predict(self, 
                   X: torch.Tensor, 
                   n: int, 
                   do_p_encoder: float, 
                   do_p_decoder: float, 
                   post_process: bool = False,
                   show_progress: bool = False,
                   ) -> np.ndarray:
        
        self.enable_dropout()
        self.do_encoder.p = do_p_encoder
        self.do_decoder.p = do_p_decoder
        
        all_out = []

        with torch.no_grad():
            if self.post_processor is not None:
                CT_ref = X[0,0,:,:,:].cpu()
            
            for i in tqdm(range(n), disable=not show_progress):
                output = self(X)
                output = output.softmax(dim=1)
                output = output[0,1,:,:,:].cpu()
                
                
                if post_process:
                    output = self.post_processor(output, CT_ref, **self.post_proc_kwargs)
            
                output = all_out.append(output.numpy())
                
            
        all_out = np.stack(all_out)
        
        self.disable_dropout()
        self.do_encoder.p = self.do_p_encoder
        self.do_decoder.p = self.do_p_decoder
        
        return all_out
    
    def MC_std(self,
                mc_out: np.ndarray, 
                pred: np.ndarray, 
                num_iteration: int = 4):    
        
        pred_dil = slicedDilationOrErosion(input_mask = pred, num_iteration = num_iteration, operation = 'dilate')
        
        mc_std = mc_out.std(axis = 0)
        
        return mc_std[pred_dil == 1].mean()
    
    def MC_fleissKappa(self, mc_out, CT_in = None, post_process = True, chunks = False):
        '''
        

        Args:
            mc_out (TYPE): DESCRIPTION.
            CT_in (TYPE, optional): DESCRIPTION. Defaults to None.
            post_process (TYPE, optional): DESCRIPTION. Defaults to True.
            chunks (int, optional): size of the chunks to run at a time to reduce afterward.

        Returns:
            TYPE: DESCRIPTION.

        '''
        if post_process:
            CT_in = CT_in[0,0,:,:,:].cpu().numpy()
            processed_mc_out = np.zeros_like(mc_out, dtype=np.int8)
            for i in range(mc_out.shape[0]):
                processed_mc_out[i,:,:,:] = self.post_processor(mc_out[i,:,:,:], CT_in, **self.post_proc_kwargs) 
            mc_out = processed_mc_out
            
        if chunks:
            all_fk = []
            for i in range(mc_out.shape[0] // chunks):
                all_fk.append(FleissKappa(mc_out[i*chunks : (i+1)*chunks]))    
            
            return np.median(all_fk)
            
        else:
            return FleissKappa(mc_out)
    
    
    def MC_entropy(self, 
                   mc_out: np.ndarray, 
                   CT_in: Optional[torch.Tensor] = None,
                   reduction: str = 'mean', 
                   method: str = 'full',
                   masking: bool = False):
        
        
        # entropy of outputs
        if method == 'entropy':
            entropy = -(mc_out*np.log(mc_out+1e-6) + (1-mc_out)*np.log(1-mc_out+1e-6))
        elif method == 'cross-entropy':
            with torch.no_grad():
                prediction = self(CT_in.cuda()).cpu().numpy()
                entropy = -(prediction[0,1,:,:,:]*np.log(mc_out+1e-6) + (prediction[0,0,:,:,:])*np.log(1-mc_out+1e-6))
        elif method == 'cross-entropy-binary':
            with torch.no_grad():
                # binarize the prediction mask
                prediction = self(CT_in.cuda()).cpu()
                prediction = self.post_processor(prediction[0,1,:,:,:].numpy(), CT_in[0,0,:,:,:].cpu().numpy(), **self.post_proc_kwargs) 
                entropy = -(prediction*np.log(mc_out+1e-6) + (1-prediction)*np.log(1-mc_out+1e-6))
        else:
            raise ValueError('Methods was not found. Options are: \'entropy\', \'cross-entropy\' or \'cross-entropy-binary\'.')
    
    
        if reduction == 'mean':
            entropy = np.mean(entropy, axis=0)
        elif reduction == 'std':
            entropy = np.std(entropy, axis=0)
        else:
            raise ValueError('Reduction was not found. Options are: \'mean\' or \'std\'.')
            
        # use dilated prediction mask for focusing the metrics
        if masking:
            with torch.no_grad():
                # binarize the prediction mask
                mask = self(CT_in.cuda()).cpu()
                mask = self.post_processor(mask[0,1,:,:,:].numpy(), CT_in[0,0,:,:,:].cpu().numpy(), **self.post_proc_kwargs) 
                # Normalize to the size of the mask before dilation
                norm_factor = mask.sum()
                mask = slicedDilationOrErosion(input_mask=mask, num_iteration=4, operation = 'dilate')
                
                entropy = entropy[mask]
                
            return np.sum(entropy)/norm_factor
        else:            
            return np.mean(entropy)

    
def FleissKappa(*args):
    '''
        Takes a variable number of predictions and one hot encodes and sum the votes, and return the
        fleiss Kappa. Works only for 2 classes (0 and 1).
        
        *args:      Sequence of scans to calculate the Fleiss Kappa for
    '''
    def one_hot(X):
        tmp = np.zeros(X.shape + (2,), dtype=np.int8)
        tmp[X == 0, 0] = 1 
        tmp[X == 1, 1] = 1 
        return tmp
    
    def aggregate_annotators(*args):
        agg_out = np.zeros_like(one_hot(args[0]), dtype=np.int8)
        for X in args:
            agg_out += one_hot(X)
    
        return agg_out
    
    def aggregate_annotators_arr(args):
        # args is an array here with the first dim being the voters
        # The last dim becomes the one hot dim with one_hot
        agg_out = np.zeros_like(one_hot(args[0]), dtype=np.int8)
        
        for i in range(args.shape[0]):
            agg_out += one_hot(args[i])
    
        return agg_out

    # args is a sequence of the inputs
    if len(args) > 1:
        agg_out = aggregate_annotators(*args)
    # args is an array with the first dim being voters
    elif len(args) == 1:
        agg_out = aggregate_annotators_arr(args[0])
        
    kappa = fleiss_kappa(agg_out.reshape(-1,2))

    return kappa



def slicedDilationOrErosion(input_mask, num_iteration, operation):
    '''
    Perform the dilation on the smallest slice that will fit the
    segmentation
    '''
    margin = 2 if num_iteration is None else num_iteration+1
    
    # find the minimum volume enclosing the organ
    x_idx = np.where(input_mask.sum(axis=(1,2)))[0]
    x_start, x_end = x_idx[0]-margin, x_idx[-1]+margin
    y_idx = np.where(input_mask.sum(axis=(0,2)))[0]
    y_start, y_end = y_idx[0]-margin, y_idx[-1]+margin
    z_idx = np.where(input_mask.sum(axis=(0,1)))[0]
    z_start, z_end = z_idx[0]-margin, z_idx[-1]+margin
    
    struct = scipy.ndimage.generate_binary_structure(3,1)
    struct = scipy.ndimage.iterate_structure(struct, num_iteration)
    
    if operation == 'dilate':
        mask_slice = scipy.ndimage.binary_dilation(input_mask[x_start:x_end, y_start:y_end, z_start:z_end], structure=struct).astype(np.int8)
    elif operation == 'erode':
        mask_slice = scipy.ndimage.binary_erosion(input_mask[x_start:x_end, y_start:y_end, z_start:z_end], structure=struct).astype(np.int8)
        
    output_mask = input_mask.copy()
    
    output_mask[x_start:x_end, y_start:y_end, z_start:z_end] = mask_slice
    
    return output_mask

class UNet3D(Abstract3DUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     **kwargs)


class ResidualUNet3D(Abstract3DUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1, 
                 set_last_bias=True, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ExtResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_padding=conv_padding,
                                             set_last_bias=set_last_bias,
                                             **kwargs)


class UNet2D(Abstract3DUNet):
    """
    Just a standard 2D Unet. Arises naturally by specifying conv_kernel_size=(1, 3, 3), pool_kernel_size=(1, 2, 2).
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        if conv_padding == 1:
            conv_padding = (0, 1, 1)
        super(UNet2D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_kernel_size=(1, 3, 3),
                                     pool_kernel_size=(1, 2, 2),
                                     conv_padding=conv_padding,
                                     **kwargs)


def get_model(config):
    def _model_class(class_name):
        m = importlib.import_module('pytorch3dunet.unet3d.model')
        clazz = getattr(m, class_name)
        return clazz

    assert 'model' in config, 'Could not find model configuration'
    model_config = config['model']
    model_class = _model_class(model_config['name'])
    return model_class(**model_config)
