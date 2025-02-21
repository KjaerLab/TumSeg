#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Callable, Optional
import torch
import torch.nn as nn
import scipy 
import torchio as tio
import nibabel as nib
import numpy as np
import os 
import joblib
import json

from unet3d.model import UNet3D
from tumseg_misc import computeRates, precisionRecallFscore, locateMice

class TumSeg(nn.Module):
    def __init__(self, net_path, device=None):
        super().__init__()
    
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.net_all = UNet3D(in_channels=1, out_channels=2, layer_order='cbr', final_sigmoid=False, 
                      set_last_bias=False)
        self.net_all.load_state_dict(torch.load(net_path))
        
        self.net_all = self.net_all.to(self.device)
        self.net_all.eval()
        
        # load the UQ model
        self.loadUQModel()
        
    def forward(self, X):
        return self.net_all(X)
    
    def init_ensemble(self, net_path_A, net_path_B, net_path_C):
        self.net_A = UNet3D(in_channels=1, out_channels=2, layer_order='cbr', final_sigmoid=False, 
                      set_last_bias=False)
        self.net_A.load_state_dict(torch.load(net_path_A))
        self.net_A = self.net_A.to(self.device)
        self.net_A.eval()
        
        self.net_B = UNet3D(in_channels=1, out_channels=2, layer_order='cbr', final_sigmoid=False, 
                      set_last_bias=False)
        self.net_B.load_state_dict(torch.load(net_path_B))
        self.net_B = self.net_B.to(self.device)
        self.net_B.eval()

        self.net_C = UNet3D(in_channels=1, out_channels=2, layer_order='cbr', final_sigmoid=False, 
                      set_last_bias=False)
        self.net_C.load_state_dict(torch.load(net_path_C))
        self.net_C = self.net_C.to(self.device)
        self.net_C.eval()
    
    def attach_post_processor(self, 
                              post_processor: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                              post_proc_kwargs: Optional[dict] = None):
        self.post_processor = post_processor
        self.post_proc_kwargs = post_proc_kwargs
        
    def attach_post_processor_UQ(self, 
                              post_processor: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                              post_proc_kwargs: Optional[dict] = None):
        
        self.net_all.post_processor = post_processor
        self.net_all.post_proc_kwargs = post_proc_kwargs
        
    def post_process(self, output, CT_in):
        return self.post_processor(output, CT_in, **self.post_proc_kwargs)
    
    def loadUQModel(self):
        self.tumseg_uq_model = joblib.load('./uq_regression_pipeline.joblib')
    
    def runUQ(self, subj, n_dropouts = 100, e_p = 0.35, d_p = 0.35, mice_slices = None):
        
        with torch.no_grad():
            CT_in = subj['CT']['data'].unsqueeze(dim=0)
            
            if mice_slices:
                predicted_dice = []
                
                for i, slice_ in enumerate(mice_slices.values()):
                    print(f'Estimating Dice for mouse: {i+1}')
                    
                    sub_ct = CT_in[:,:,slice_['y_range'][0]:slice_['y_range'][1], 
                                       slice_['x_range'][0]:slice_['x_range'][1],:]
                    
                    predicted_dice.append(self.predictDice(sub_ct, n_dropouts, e_p, d_p))
                
                return predicted_dice
            else:
                return [self.predictDice(CT_in, n_dropouts, e_p, d_p)]
            
    def predictDice(self, CT_in, n_dropouts, e_p, d_p):   
          
        do_out = self.net_all.MC_predict(CT_in.to(self.device), n=n_dropouts, 
                            do_p_encoder=e_p, do_p_decoder=d_p, post_process=False, show_progress = True)
        
        
        entropy_masked = self.net_all.MC_entropy(do_out, CT_in = CT_in, reduction='mean', method='entropy',
                                  masking = True)
    
        cross_entropy_binary_masked = self.net_all.MC_entropy(do_out,CT_in = CT_in, reduction='mean', method='cross-entropy-binary',
                                  masking = True)
        
        fleiss_kappa = self.net_all.MC_fleissKappa(do_out, CT_in=CT_in, post_process = True, chunks = 10)
        
        # Run the ensemble 
        output_A = self.net_A(CT_in.to(self.device))
        output_A = output_A.softmax(dim=1)
        output_A = output_A[0,1,:,:,:].detach().cpu()
        output_A = self.post_process(output_A, CT_in[0,0])            
                    
        output_B = self.net_A(CT_in.to(self.device))
        output_B = output_B.softmax(dim=1)
        output_B = output_B[0,1,:,:,:].detach().cpu()
        output_B = self.post_process(output_B, CT_in[0,0])                    
        
        output_C = self.net_A(CT_in.to(self.device))
        output_C = output_C.softmax(dim=1)
        output_C = output_C[0,1,:,:,:].detach().cpu()
        output_C = self.post_process(output_C, CT_in[0,0])            
        
        dice_ensemble = np.mean(getAllF1Combinations(output_A,output_B,output_C))
            
        X_uq = np.transpose(np.vstack([fleiss_kappa, entropy_masked, cross_entropy_binary_masked, dice_ensemble]))
        
        return self.tumseg_uq_model.predict(X_uq)[0]
    
    

def getAllF1Combinations(roi_a, roi_b, roi_c, reference=None):
    tmp_f1 = []
    
    if reference is None:
        TP, TN, FP, FN = computeRates(roi_a, roi_b, thress=0.5)
        precision, recall, f1_score = precisionRecallFscore(TP, TN, FP, FN )
        tmp_f1.append(f1_score.numpy())
        
        TP, TN, FP, FN = computeRates(roi_a, roi_c, thress=0.5)
        precision, recall, f1_score = precisionRecallFscore(TP, TN, FP, FN )
        tmp_f1.append(f1_score.numpy())
        
        TP, TN, FP, FN = computeRates(roi_b, roi_c, thress=0.5)
        precision, recall, f1_score = precisionRecallFscore(TP, TN, FP, FN )
        tmp_f1.append(f1_score.numpy())
    else:
        TP, TN, FP, FN = computeRates(roi_a, reference, thress=0.5)
        precision, recall, f1_score = precisionRecallFscore(TP, TN, FP, FN )
        tmp_f1.append(f1_score.numpy())
        
        TP, TN, FP, FN = computeRates(roi_b, reference, thress=0.5)
        precision, recall, f1_score = precisionRecallFscore(TP, TN, FP, FN )
        tmp_f1.append(f1_score.numpy())
        
        TP, TN, FP, FN = computeRates(roi_c, reference, thress=0.5)
        precision, recall, f1_score = precisionRecallFscore(TP, TN, FP, FN )
        tmp_f1.append(f1_score.numpy())
    
    return tmp_f1


def windowCT(min_=-400, max_=400):
    '''
    Helper function for the transform pipeline
    '''    
    def windowCT_(subject):
        subject['CT']['data'][subject['CT']['data'] < min_] = min_ 
        subject['CT']['data'][subject['CT']['data'] > max_] = max_ 
        
        return subject
    
    return windowCT_

def nifti_loader_float(path):
    img_prox = nib.load(path)
    # return torch.Tensor(img_prox.get_fdata()).type(torch.FloatTensor), np.eye(4)
    return torch.Tensor(img_prox.get_fdata()).type(torch.FloatTensor), img_prox.header.get_base_affine()

def buildSubjectList(input_path):
    subjects = []
    # input is a single nifti file
    if input_path.endswith('nii') or input_path.endswith('nii.gz'):
        ct_path = input_path
        subjects.append(tio.Subject(
            CT = tio.ScalarImage(ct_path, reader=nifti_loader_float),
            CT_prox = nib.load(ct_path),
            path = ct_path
        )
            )
    # input is a folder
    else:
        for path, folders, files in os.walk(input_path):
            if files:
                for file in files:
                    if file.endswith('nii') or file.endswith('nii.gz'):
                        ct_path = os.path.join(path,file)
                        subjects.append(tio.Subject(
                            CT = tio.ScalarImage(ct_path, reader=nifti_loader_float),
                            CT_prox = nib.load(ct_path),
                            path = ct_path
                        )
                            )
        
    return subjects

def runInference(subj, tumseg, mice_slices):
    with torch.no_grad():
        
        CT_in = subj['CT']['data'].unsqueeze(dim=0)
     
        if mice_slices:
            output = torch.zeros(CT_in.shape[2:])    
            for i, slice_ in enumerate(mice_slices.values()):
                print(f'Predicting mouse: {i+1}')
                
                sub_ct = CT_in[:,:,slice_['y_range'][0]:slice_['y_range'][1], 
                                   slice_['x_range'][0]:slice_['x_range'][1],:]
                
                sub_output = tumseg(sub_ct.to(tumseg.device))
                sub_output = sub_output.softmax(dim=1)
                
                output[slice_['y_range'][0]:slice_['y_range'][1],
                       slice_['x_range'][0]:slice_['x_range'][1],:] = sub_output[0,1,:,:,:].detach().cpu()
            
        else:    
            output = tumseg(CT_in.to(tumseg.device))
            output = output.softmax(dim=1)
            
            output = output[0,1,:,:,:].detach().cpu()
        
    return output

def resampleAndPostProcess(output, subj, tumseg, target_pixel_size, locate_mice):
    # inverse transform and post process 
    target_inverse_zoom = np.array(target_pixel_size) / np.array(subj['CT_prox'].header.get_zooms())
    output = scipy.ndimage.zoom(output, zoom=target_inverse_zoom, order=1) > 0.5 
    
    CT_in = nib.load(subj['path']).get_fdata()
    CT_in = np.clip(CT_in, -400, 400)
    post_processed = tumseg.post_process(output, CT_in)
    
    # locate the since the CT is loaded now 
    if locate_mice:
        # decide the size of the bouding box
        crop_factor = np.max(target_inverse_zoom[:2])
        crop_size = int(112 * crop_factor)
        
        mice_slices = locateMice(CT_in, crop_size=[crop_size, crop_size], debug=False, verbose=False)
    else:
        mice_slices = None
    
    output = {
        'post_processed': post_processed,
        'mice_slices': mice_slices
        }
    
    return output

def saveResults(output, subj, input_path, output_path, uq_pred=None):
    # Save results as a nifti 
    if input_path.endswith('nii.gz') or input_path.endswith('nii'):
        input_path = '/'.join(input_path.split(os.sep)[:-1])
                     
    rel_path = os.path.relpath('/'.join(subj['path'].split(os.sep)[:-1]), input_path)

    if output_path is None:
        output_path = input_path 
    
    os.makedirs(os.path.join(output_path, rel_path), exist_ok=True)
    
    # make new nifti file
    new_nii = nib.Nifti1Image(output['post_processed'].astype(np.int8), subj['CT_prox'].affine, subj['CT_prox'].header)
    
    # create the save path
    scan_name = subj['path'].split(os.sep)[-1].split('.')[0]
    full_save_path = os.path.join(output_path, rel_path, scan_name + '_tumor_mask.nii.gz')
    
    print('Saving results to: ' + full_save_path)
    nib.save(new_nii, full_save_path)
    
    if uq_pred is not None:
        with open(os.path.join(output_path, rel_path, 'uq_dice_prediction.txt'), 'w') as f:
            for i, dice in enumerate(uq_pred):
                f.write('Mouse_{},{:.3f}\n'.format(i+1, dice))
    
    if output['mice_slices']:
        with open(os.path.join(output_path, rel_path, 'mice_bounding_box.json'), "w") as f: 
            json.dump(output['mice_slices'], f, indent=4)
    
         
    
    
